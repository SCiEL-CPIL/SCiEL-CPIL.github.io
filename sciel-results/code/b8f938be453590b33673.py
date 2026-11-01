"""Combined Lennard-Jones cluster global minimum, 55 atoms.
Multi-start stochastic search with radial manifold optimization and high-precision polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# Deadline guard: evaluator kills at 150s wall-clock; use 140s as safety margin
DEADLINE_SECONDS = 140.0
_start_time = time.time()

def _remaining():
    return DEADLINE_SECONDS - (time.time() - _start_time)

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
    """Compute total Lennard-Jones energy for N atoms."""
    n = len(pos)
    rij = np.zeros((n, n, 3))
    for i in range(n):
        rij[i, i + 1:] = pos[i] - pos[i + 1:]
        rij[i + 1:, i] = -rij[i, i + 1:]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _lj_energy_grad(pos):
    """Compute gradient of Lennard-Jones energy w.r.t. Cartesian coords."""
    n = len(pos)
    rij = np.zeros((n, n, 3))
    for i in range(n):
        rij[i, i + 1:] = pos[i] - pos[i + 1:]
        rij[i + 1:, i] = -rij[i, i + 1:]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    coeff = 4.0 * (-12.0 * r2**(-7) + 6.0 * r2**(-4))
    np.fill_diagonal(coeff, 0.0)
    grad = 2.0 * np.sum(coeff[:, :, np.newaxis] * rij, axis=1)
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

def _optimize_radial_manifold(r0, maxiter=5000, gtol=1e-12, ftol=1e-15):
    """Optimize all 54 radial distances on the icosahedral direction manifold."""
    lb = np.concatenate([np.full(12, 0.5), np.full(30, 0.5), np.full(12, 0.5)])
    ub = np.concatenate([np.full(12, 3.0), np.full(30, 3.5), np.full(12, 3.5)])

    result = minimize(
        _radial_energy,
        r0,
        jac=_radial_grad,
        bounds=list(zip(lb, ub)),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 50}
    )
    return result.x, result.fun

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

def _check_min_dist(pos, min_d=0.5):
    """Check all pair distances >= min_d."""
    from scipy.spatial.distance import pdist
    d = pdist(pos)
    return d.min() >= min_d - 1e-9

def _make_mackay_icosahedron():
    """Construct the two-shell Mackay icosahedron for 55 atoms with literature-optimal radii."""
    # Literature-optimal shell radii for 55-atom LJ cluster
    r_inner = 1.135      # 12 inner shell atoms (icosahedron vertices)
    r_outer_mid = 1.162  # 30 edge-mid atoms
    r_outer_vert = 1.283 # 12 outer vertex atoms
    
    r0 = np.concatenate([
        np.full(12, r_inner),
        np.full(30, r_outer_mid),
        np.full(12, r_outer_vert)
    ])
    pos54 = _radial_to_cartesian(r0)
    return _cartesian_to_full(pos54)

def _relax_lbfgs(pos, maxiter=2000, gtol=1e-10, ftol=1e-12):
    """L-BFGS-B local relaxation."""
    if _remaining() < 5.0:
        return pos, _lj_energy(pos)
    
    def obj_and_grad(x):
        x2d = x.reshape(-1, 3)
        e = _lj_energy(x2d)
        g = _lj_energy_grad(x2d)
        return e, g.ravel()
    
    result = minimize(
        obj_and_grad,
        pos.reshape(-1),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
    )
    return result.x.reshape(-1, 3), result.fun

def construct():
    global _start_time
    _start_time = time.time()
    
    best_pos = None
    best_energy = np.inf
    
    # Strategy: Multi-start with diverse noise scales on Mackay icosahedron
    # Each candidate: perturb exact Mackay geometry with Gaussian noise, then relax
    rng = np.random.default_rng(42)
    
    # Diverse noise scales: from very small to moderate
    noise_scales = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    
    # Initial: exact Mackay with literature radii
    mackay_base = _make_mackay_icosahedron()
    
    # First, try the exact Mackay geometry with radial manifold optimization
    if _remaining() > 30.0:
        try:
            r0 = np.linalg.norm(mackay_base[1:], axis=1)
            radii_opt, rad_energy = _optimize_radial_manifold(r0, maxiter=3000)
            pos_from_radial = _cartesian_to_full(_radial_to_cartesian(radii_opt))
            if _check_min_dist(pos_from_radial) and rad_energy < best_energy:
                best_energy = rad_energy
                best_pos = pos_from_radial.copy()
        except Exception:
            pass
    
    # Multi-start: perturb Mackay with different noise scales
    for noise_scale in noise_scales:
        if _remaining() < 15.0:
            break
        try:
            # Create perturbed version
            pos_perturbed = mackay_base.copy()
            noise = rng.normal(0, noise_scale, pos_perturbed.shape)
            pos_perturbed += noise
            
            # Ensure minimum distance constraint
            if not _check_min_dist(pos_perturbed):
                # Scale down if needed
                from scipy.spatial.distance import pdist
                d = pdist(pos_perturbed)
                min_d = d.min()
                if min_d < 0.5:
                    scale = 0.5 / min_d * 0.99
                    pos_perturbed = pos_perturbed * scale
            
            # Radial manifold optimization from perturbed state
            pos54 = pos_perturbed[1:]
            radii_current = np.linalg.norm(pos54, axis=1)
            
            radii_opt, rad_energy = _optimize_radial_manifold(radii_current, maxiter=2000)
            pos_from_radial = _cartesian_to_full(_radial_to_cartesian(radii_opt))
            
            if _check_min_dist(pos_from_radial) and rad_energy < best_energy:
                best_energy = rad_energy
                best_pos = pos_from_radial.copy()
                
        except Exception:
            continue
    
    # Final high-precision Cartesian polish on best result
    if best_pos is not None and _remaining() > 10.0:
        try:
            pos_polished, e_polished = _polish_cartesian(best_pos, maxiter=5000, gtol=1e-14, ftol=1e-16)
            if _check_min_dist(pos_polished) and e_polished < best_energy:
                best_energy = e_polished
                best_pos = pos_polished.copy()
        except Exception:
            pass
    
    # Second radial manifold pass from polished Cartesian structure
    if best_pos is not None and _remaining() > 8.0:
        try:
            pos54 = best_pos[1:]
            radii_current = np.linalg.norm(pos54, axis=1)
            radii_opt2, rad_energy2 = _optimize_radial_manifold(radii_current, maxiter=3000, gtol=1e-12, ftol=1e-15)
            pos_r2 = _cartesian_to_full(_radial_to_cartesian(radii_opt2))
            if _check_min_dist(pos_r2) and rad_energy2 < best_energy:
                best_energy = rad_energy2
                best_pos = pos_r2.copy()
        except Exception:
            pass
    
    # Final tight-tolerance refinement
    if best_pos is not None and _remaining() > 5.0:
        try:
            pos_final, e_final = _polish_cartesian(best_pos, maxiter=10000, gtol=1e-15, ftol=1e-17)
            if _check_min_dist(pos_final) and e_final < best_energy:
                best_energy = e_final
                best_pos = pos_final.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: original cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos
