"""Improved program: low-dimensional radial manifold optimization for 55-atom Mackay icosahedron,
then high-precision L-BFGS polish with deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# Deadline guard: evaluator kills at 150s wall-clock; use 115s as safety margin
DEADLINE_SECONDS = 115.0

# Precompute icosahedral directions
_PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _get_icosahedral_directions():
    """Return 54 unit direction vectors: 12 vertex dirs, 30 edge-mid dirs, 12 vertex dirs again."""
    verts = np.array([
        [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
        [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
        [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms

    # Find edges
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
    edge_mids = np.array(edge_mids)  # (30, 3)

    # Order: 12 inner-vertex dirs, 30 edge-mid dirs, 12 outer-vertex dirs
    dirs = np.vstack([verts_unit, edge_mids, verts_unit])
    return dirs  # shape (54, 3)

_DIRS = _get_icosahedral_directions()  # precomputed at module load

def _radial_to_cartesian(radii):
    """Convert 54 radial distances to 54 Cartesian positions (excluding center at origin)."""
    return radii[:, np.newaxis] * _DIRS  # shape (54, 3)

def _cartesian_to_full(pos54):
    """Add center atom at origin to make full 55-atom array."""
    return np.vstack([np.zeros(3), pos54])

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for N atoms (vectorized)."""
    n = len(pos)
    # Use broadcasting: pos[:, None, :] - pos[None, :, :] gives (n, n, 3)
    rij = pos[:, None, :] - pos[None, :, :]  # (n, n, 3)
    r2 = np.sum(rij**2, axis=2)  # (n, n)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _lj_energy_grad(pos):
    """Compute gradient of Lennard-Jones energy w.r.t. Cartesian coords (vectorized)."""
    n = len(pos)
    rij = pos[:, None, :] - pos[None, :, :]  # (n, n, 3)
    r2 = np.sum(rij**2, axis=2)  # (n, n)
    np.fill_diagonal(r2, 1.0)
    # dE/dr2 = 4 * (-12/r2^7 + 6/r2^4) = 4 * (-12*r2^-7 + 6*r2^-4)
    # dE/dx_i = sum_j 2 * dE/dr2_ij * (x_i - x_j)
    coeff = 4.0 * (-12.0 * r2**(-7) + 6.0 * r2**(-4))  # (n, n)
    np.fill_diagonal(coeff, 0.0)
    grad = 2.0 * np.sum(coeff[:, :, np.newaxis] * rij, axis=1)  # (n, 3)
    return grad

def _radial_energy(radii):
    """Energy as function of 54 radial distances (center fixed at origin)."""
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    return _lj_energy(pos_full)

def _radial_grad(radii):
    """Gradient of energy w.r.t. radial distances via chain rule."""
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    cart_grad = _lj_energy_grad(pos_full)  # shape (55, 3)
    # grad w.r.t. radii[i] = dot(cart_grad[i+1], dir[i])
    grad = np.sum(cart_grad[1:] * _DIRS, axis=1)
    return grad

def _optimize_radial_manifold():
    """Optimize all 54 radial distances on the icosahedral direction manifold.
    Uses Nelder-Mead for global exploration followed by L-BFGS-B for local refinement."""
    # Initial guess: standard Mackay radii
    r_inner = 1.054
    r_outer_mid = 1.755
    r_outer_vert = 2.0
    r0 = np.concatenate([
        np.full(12, r_inner),
        np.full(30, r_outer_mid),
        np.full(12, r_outer_vert)
    ])

    # Bounds: ensure minimum separation >= 0.5
    lb = np.concatenate([np.full(12, 0.5), np.full(30, 0.5), np.full(12, 0.5)])
    ub = np.concatenate([np.full(12, 3.0), np.full(30, 3.5), np.full(12, 3.5)])

    # Phase 1: Nelder-Mead for global exploration on radial manifold
    # Use moderate tolerances to avoid excessive iterations
    nm_result = minimize(
        _radial_energy,
        r0,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
    )
    
    # Clip to bounds
    x_nm = np.clip(nm_result.x, lb, ub)
    e_nm = _radial_energy(x_nm)
    
    # Phase 2: L-BFGS-B local refinement from Nelder-Mead result
    lbfgs_result = minimize(
        _radial_energy,
        x_nm,
        jac=_radial_grad,
        bounds=list(zip(lb, ub)),
        method='L-BFGS-B',
        options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-15, 'maxls': 50}
    )
    
    return lbfgs_result.x, lbfgs_result.fun

def _polish_cartesian(pos0, maxiter=10000, gtol=1e-14, ftol=1e-16):
    """High-precision L-BFGS polish in full 165D Cartesian space."""
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))

    def grad_func(x):
        return _lj_energy_grad(x.reshape(-1, 3)).flatten()

    result = minimize(
        energy_func,
        pos0.flatten(),
        jac=grad_func,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 50}
    )
    return result.x.reshape(-1, 3), result.fun

def construct():
    start_time = time.time()
    best_pos = None
    best_energy = np.inf

    # Step 1: Optimize on the 54D radial manifold (Nelder-Mead + L-BFGS-B)
    try:
        radii_opt, rad_energy = _optimize_radial_manifold()
        pos_from_radial = _cartesian_to_full(_radial_to_cartesian(radii_opt))
        best_pos = pos_from_radial
        best_energy = rad_energy
    except Exception:
        # Fallback: use standard Mackay radii
        r_inner, r_outer_mid, r_outer_vert = 1.054, 1.755, 2.0
        r0 = np.concatenate([
            np.full(12, r_inner),
            np.full(30, r_outer_mid),
            np.full(12, r_outer_vert)
        ])
        best_pos = _cartesian_to_full(_radial_to_cartesian(r0))
        best_energy = _lj_energy(best_pos)

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 20:
        return best_pos

    # Step 2: High-precision Cartesian polish
    try:
        pos_polished, e_polished = _polish_cartesian(best_pos, maxiter=10000, gtol=1e-14, ftol=1e-16)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pos = pos_polished
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 12:
        return best_pos

    # Step 3: Second radial manifold pass from polished Cartesian structure
    try:
        pos54 = best_pos[1:]  # exclude center
        radii_current = np.linalg.norm(pos54, axis=1)
        
        lb = np.concatenate([np.full(12, 0.5), np.full(30, 0.5), np.full(12, 0.5)])
        ub = np.concatenate([np.full(12, 3.0), np.full(30, 3.5), np.full(12, 3.5)])
        
        result2 = minimize(
            _radial_energy,
            radii_current,
            jac=_radial_grad,
            bounds=list(zip(lb, ub)),
            method='L-BFGS-B',
            options={'maxiter': 3000, 'gtol': 1e-12, 'ftol': 1e-15, 'maxls': 50}
        )
        pos_r2 = _cartesian_to_full(_radial_to_cartesian(result2.x))
        e_r2 = result2.fun
        if e_r2 < best_energy:
            best_energy = e_r2
            best_pos = pos_r2
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 8:
        return best_pos

    # Step 4: Final ultra-precise polish
    try:
        pos_final, e_final = _polish_cartesian(best_pos, maxiter=5000, gtol=1e-15, ftol=1e-17)
        if e_final < best_energy:
            best_energy = e_final
            best_pos = pos_final
    except Exception:
        pass

    # Step 5: Verify minimum separation constraint; if violated, project back
    try:
        pos_check = best_pos.copy()
        n = len(pos_check)
        rij = pos_check[:, None, :] - pos_check[None, :, :]
        r2 = np.sum(rij**2, axis=2)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(np.min(r2))
        if min_r < 0.5:
            # Nudge atoms apart: scale all radii by 0.5/min_r
            scale = 0.5 / min_r * 1.01
            pos_check[1:] *= scale
            e_check = _lj_energy(pos_check)
            if e_check < best_energy:
                best_pos = pos_check
                best_energy = e_check
            else:
                # Try a small Cartesian polish from the nudged position
                try:
                    pos_nudged, e_nudged = _polish_cartesian(pos_check, maxiter=2000, gtol=1e-12, ftol=1e-14)
                    if e_nudged < best_energy:
                        best_pos = pos_nudged
                        best_energy = e_nudged
                except Exception:
                    pass
    except Exception:
        pass

    return best_pos
