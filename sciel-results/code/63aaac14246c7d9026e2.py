"""Lennard-Jones 55-atom cluster: global optimization via Mackay icosahedron manifold."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE = 142.0  # Safety margin before 150s kill


def lj_energy(pos):
    """Compute total LJ energy for N atoms (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e


def lj_energy_grad(pos):
    """Compute LJ energy and gradient (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad


def _get_icosahedron_geometry():
    """
    Returns the 12 icosahedron vertex unit vectors and 30 edge-midpoint unit vectors.
    The outer shell uses the same directions as the inner shell.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    # Find edges
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))

    # 30 edge midpoints (normalized)
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)

    return verts, midpoints


# Precompute geometry
VERTS, MIDPOINTS = _get_icosahedron_geometry()


def _make_rotation_matrix(euler):
    """Create a rotation matrix from Euler angles (xyz convention)."""
    rx, ry, rz = euler
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def _make_mackay_from_params(params):
    """
    Generate Mackay icosahedron positions from parameters.
    params: [r1, r2, r3, rx, ry, rz]
    r1: radius for 12 inner vertices (shell 1)
    r2: radius for 30 edge midpoints (shell 2)
    r3: radius for 12 outer vertices (shell 3, same directions as shell 1)
    """
    r1, r2, r3 = np.maximum(params[:3], 0.5)
    R = _make_rotation_matrix(params[3:6])

    pos = np.zeros((55, 3))
    # Atom 0: center
    pos[0] = [0.0, 0.0, 0.0]

    # Atoms 1-12: Shell 1 (icosahedron vertices)
    for i in range(12):
        pos[1 + i] = R @ (VERTS[i] * r1)

    # Atoms 13-42: Shell 2 (edge midpoints)
    for i in range(30):
        pos[13 + i] = R @ (MIDPOINTS[i] * r2)

    # Atoms 43-54: Shell 3 (icosahedron vertices, outer, same directions as shell 1)
    for i in range(12):
        pos[43 + i] = R @ (VERTS[i] * r3)

    return pos


def relax_positions(pos, maxiter=200, gtol=1e-6):
    """Relax atomic positions using L-BFGS-B."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)

    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-10, 'maxls': 50})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)


def _objective_radial(params_radial):
    """
    Objective for optimizing the 3 radial parameters only.
    Builds the Mackay structure and computes its energy (no relaxation).
    """
    r1, r2, r3 = params_radial
    if r1 < 0.5 or r2 < 0.5 or r3 < 0.5:
        return 1e10
    pos = _make_mackay_from_params(np.array([r1, r2, r3, 0.0, 0.0, 0.0]))
    # Check minimum distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 0.5:
        return 1e10
    return lj_energy(pos)


def optimize_radial_params(params_init, maxiter_nm=100):
    """
    Optimize the 3 radial parameters using Nelder-Mead.
    """
    def objective(params):
        return _objective_radial(params)

    try:
        res = minimize(objective, params_init, method='Nelder-Mead',
                       options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True, 'maxfev': maxiter_nm * 3})
        return res.x, res.fun
    except Exception:
        return params_init, _objective_radial(params_init)


def optimize_full_manifold(params_init, maxiter_nm=50, maxiter_lbfgs=150):
    """
    Optimize manifold parameters (3 radii + 3 rotation angles) using Nelder-Mead.
    Each evaluation relaxes the full 165-dimensional position space.
    """
    def objective(params):
        pos = _make_mackay_from_params(params)
        pos_relaxed, e = relax_positions(pos, maxiter=maxiter_lbfgs, gtol=1e-6)
        return e

    try:
        res = minimize(objective, params_init, method='Nelder-Mead',
                       options={'maxiter': maxiter_nm, 'xatol': 1e-5, 'fatol': 1e-5,
                                'maxfev': maxiter_nm * 2})
        best_params = res.x
        best_pos = _make_mackay_from_params(best_params)
        best_pos, best_e = relax_positions(best_pos, maxiter=300, gtol=1e-8)
        return best_pos, best_e, best_params
    except Exception:
        return None, np.inf, params_init


def polish_high_precision(pos, deadline=None):
    """
    High-precision polishing with multiple stages of increasing strictness.
    """
    best_energy = lj_energy(pos)
    best_pos = pos.copy()

    def obj_and_grad(x):
        pos_flat = x.reshape(N, 3)
        e, g = lj_energy_grad(pos_flat)
        return e, g.flatten()

    x0 = pos.flatten()

    stages = [
        {'maxiter': 500, 'gtol': 1e-10, 'maxls': 100, 'ftol': 1e-12},
        {'maxiter': 1000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-14},
        {'maxiter': 2000, 'gtol': 1e-14, 'maxls': 300, 'ftol': 1e-16},
        {'maxiter': 3000, 'gtol': 1e-16, 'maxls': 500, 'ftol': 1e-18},
    ]

    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        try:
            result = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                              options=stage_opts)
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(N, 3)
                x0 = result.x
        except Exception:
            pass

    return best_pos, best_energy


def construct():
    start_time = time.time()

    def remaining():
        return DEADLINE - (time.time() - start_time)

    # Phase 1: Optimize the 3 radial parameters on the symmetry manifold
    # Known good starting points for LJ55 Mackay icosahedron
    radial_starts = [
        [1.04, 1.76, 1.98],   # Approx optimal
        [1.05, 1.75, 1.95],
        [1.00, 1.70, 1.90],
        [1.10, 1.80, 2.00],
        [0.95, 1.65, 1.85],
        [1.04, 1.70, 1.98],
        [1.04, 1.80, 1.98],
        [1.02, 1.72, 1.96],
    ]

    best_radial_params = None
    best_radial_energy = np.inf

    for start in radial_starts:
        if remaining() < 15:
            break
        try:
            params_opt, e_opt = optimize_radial_params(start, maxiter_nm=80)
            if e_opt < best_radial_energy:
                best_radial_energy = e_opt
                best_radial_params = params_opt
        except Exception:
            continue

    if best_radial_params is None:
        best_radial_params = np.array([1.04, 1.76, 1.98])
        best_radial_energy = _objective_radial(best_radial_params)

    # Build initial structure from optimized radial params
    pos_0 = _make_mackay_from_params(np.append(best_radial_params, [0.0, 0.0, 0.0]))

    # Phase 2: Full 3D L-BFGS relaxation
    pos_relaxed, e_relaxed = relax_positions(pos_0, maxiter=300, gtol=1e-8)

    best_pos = pos_relaxed.copy()
    best_e = e_relaxed

    # Phase 3: Try a few more full manifold optimizations with different rotations
    full_starts = [
        np.array([1.04, 1.76, 1.98, 0.0, 0.0, 0.0]),
        np.array([1.05, 1.75, 1.95, 0.0, 0.0, 0.0]),
        np.array([1.00, 1.70, 1.90, 0.0, 0.0, 0.0]),
    ]

    for params_init in full_starts:
        if remaining() < 20:
            break
        try:
            pos, e, params = optimize_full_manifold(params_init, maxiter_nm=30, maxiter_lbfgs=100)
            if pos is not None and e < best_e:
                best_e = e
                best_pos = pos.copy()
        except Exception:
            continue

    # Phase 4: Basin-Hopping with small perturbations
    rng = np.random.RandomState(42)
    bh_steps = 15
    current_pos = best_pos.copy()
    current_e = best_e

    for step in range(bh_steps):
        if remaining() < 10:
            break

        scale = 0.05 + 0.02 * step
        try:
            displacement = rng.randn(3) * scale
            perturbed = current_pos + displacement
            small_pert = rng.randn(*current_pos.shape) * (scale * 0.3)
            perturbed = perturbed + small_pert

            new_pos, new_e = relax_positions(perturbed, maxiter=100, gtol=1e-6)

            delta = new_e - current_e
            if delta < 0:
                current_pos = new_pos
                current_e = new_e
                if current_e < best_e:
                    best_e = current_e
                    best_pos = current_pos.copy()
            else:
                kT = 0.05
                if rng.rand() < np.exp(-delta / kT):
                    current_pos = new_pos
                    current_e = new_e
        except Exception:
            continue

    # Phase 5: Final high-precision polish
    if remaining() > 5:
        try:
            final_pos, final_e = polish_high_precision(best_pos, deadline=start_time + DEADLINE)
            if final_e < best_e:
                best_e = final_e
                best_pos = final_pos
        except Exception:
            pass

    return best_pos
