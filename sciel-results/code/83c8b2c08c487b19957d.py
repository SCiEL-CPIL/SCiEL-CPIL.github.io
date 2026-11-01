"""Combined: 54D radial manifold optimization for 55-atom Mackay icosahedron,
with multi-stage high-precision L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 112.0

_PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _get_icosahedral_directions():
    """Return 54 unit direction vectors: 12 vertex, 30 edge-mid, 12 vertex."""
    verts = np.array([
        [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
        [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
        [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms

    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            dists[i, j] = d
            dists[j, i] = d
    min_dist = np.min(dists[dists > 0])
    edge_mask = dists < min_dist * 1.001

    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_unit = mid / np.linalg.norm(mid)
                edge_mids.append(mid_unit)
    edge_mids = np.array(edge_mids)

    dirs = np.vstack([verts_unit, edge_mids, verts_unit])
    return dirs

_DIRS = _get_icosahedral_directions()

def _radial_to_cartesian(radii):
    return radii[:, np.newaxis] * _DIRS

def _cartesian_to_full(pos54):
    return np.vstack([np.zeros(3), pos54])

def _lj_energy(pos):
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    iu = np.triu_indices(n, k=1)
    return 4.0 * np.sum(r12[iu] - r6[iu])

def _lj_energy_grad(pos):
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    coeff = 4.0 * (-12.0 * r2**(-7) + 6.0 * r2**(-4))
    np.fill_diagonal(coeff, 0.0)
    grad = 2.0 * np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _radial_energy(radii):
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    return _lj_energy(pos_full)

def _radial_grad(radii):
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    cart_grad = _lj_energy_grad(pos_full)
    return np.sum(cart_grad[1:] * _DIRS, axis=1)

def _optimize_radial_manifold(r0=None, maxiter=20000):
    if r0 is None:
        r_inner = 1.054
        r_outer_mid = 1.755
        r_outer_vert = 2.0
        r0 = np.concatenate([
            np.full(12, r_inner),
            np.full(30, r_outer_mid),
            np.full(12, r_outer_vert)
        ])
    lb = np.concatenate([np.full(12, 0.5), np.full(30, 0.5), np.full(12, 0.5)])
    ub = np.concatenate([np.full(12, 3.0), np.full(30, 3.5), np.full(12, 3.5)])
    result = minimize(
        _radial_energy, r0, jac=_radial_grad,
        bounds=list(zip(lb, ub)),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': 1e-16, 'ftol': 1e-18, 'maxls': 200}
    )
    return result.x, result.fun

def _multi_stage_polish(pos0, deadline):
    """Multi-stage L-BFGS polish with progressively tighter tolerances."""
    best_pos = pos0.copy()
    best_energy = _lj_energy(pos0)
    x0 = pos0.flatten()

    stages = [
        {'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-15, 'maxls': 200},
        {'maxiter': 10000, 'gtol': 1e-14, 'ftol': 1e-16, 'maxls': 500},
        {'maxiter': 20000, 'gtol': 1e-16, 'ftol': 1e-18, 'maxls': 1000},
    ]

    for stage_opts in stages:
        if time.time() > deadline - 5.0:
            break
        try:
            result = minimize(
                lambda x: _lj_energy(x.reshape(-1, 3)),
                x0,
                jac=lambda x: _lj_energy_grad(x.reshape(-1, 3)).flatten(),
                method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy - 1e-15:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
            else:
                break
        except Exception:
            break

    return best_pos, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Step 1: Radial manifold optimization with very tight tolerance
    try:
        radii_opt, rad_energy = _optimize_radial_manifold(maxiter=30000)
        best_pos = _cartesian_to_full(_radial_to_cartesian(radii_opt))
        best_energy = rad_energy
    except Exception:
        r0 = np.concatenate([
            np.full(12, 1.054), np.full(30, 1.755), np.full(12, 2.0)
        ])
        best_pos = _cartesian_to_full(_radial_to_cartesian(r0))
        best_energy = _lj_energy(best_pos)

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 15:
        return best_pos

    # Step 2: Multi-stage Cartesian polish with progressively tighter tolerances
    try:
        best_pos, best_energy = _multi_stage_polish(best_pos, deadline)
    except Exception:
        pass

    return best_pos
