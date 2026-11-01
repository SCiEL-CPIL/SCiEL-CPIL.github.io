"""Improved program: construct the 55-atom Mackay icosahedron, optimize on a
low-dimensional radial manifold (54 radii on fixed icosahedral directions),
then apply full L-BFGS relaxation with a deadline guard.
Includes a high-precision polish pass."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# Deadline guard: evaluator kills at 150s wall-clock; use 110s as safety margin
DEADLINE_SECONDS = 110.0

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for N atoms (vectorized)."""
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    return 4.0 * np.sum(r12 - r6)

def _lj_energy_grad(pos):
    """Compute gradient of Lennard-Jones energy (vectorized)."""
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    coeff = 4.0 * (-12.0 * r2**(-7) + 6.0 * r2**(-4))
    np.fill_diagonal(coeff, 0.0)
    grad = 2.0 * np.sum(coeff[:, :, np.newaxis] * rij, axis=1)
    return grad

def _get_mackay_directions():
    """Precompute the 55 unit direction vectors for the Mackay icosahedron geometry.
    Returns (verts_unit, edge_midpoints_unit) where:
      verts_unit: (12, 3) unit vectors for icosahedron vertices
      edge_midpoints_unit: (30, 3) unit vectors for edge midpoints
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    rij = verts_unit[:, np.newaxis, :] - verts_unit[np.newaxis, :, :]
    dists = np.sqrt(np.sum(rij**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    edge_mask = dists < min_dist * 1.001
    
    edge_midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_unit = mid / np.linalg.norm(mid)
                edge_midpoints.append(mid_unit)
    
    edge_midpoints = np.array(edge_midpoints)
    return verts_unit, edge_midpoints

# Precompute directions once
_VERTS_UNIT, _EDGE_MIDS_UNIT = _get_mackay_directions()

def _mackay_icosahedron_55(r_inner, r_edge_mid, r_vertex):
    """
    Construct 55-atom Mackay icosahedron with three independent radii.
    """
    points = [np.zeros(3)]
    for v in _VERTS_UNIT:
        points.append(r_inner * v)
    for v in _EDGE_MIDS_UNIT:
        points.append(r_edge_mid * v)
    for v in _VERTS_UNIT:
        points.append(r_vertex * v)
    return np.array(points, dtype=float)

def _get_all_directions():
    """Get all 54 non-center direction vectors (12 inner + 30 mid + 12 outer)."""
    return np.vstack([_VERTS_UNIT, _EDGE_MIDS_UNIT, _VERTS_UNIT])

def _radial_to_cartesian(radials):
    """Convert 54 radial coordinates to 55 Cartesian positions (center at origin)."""
    dirs = _ALL_DIRS
    pos = np.zeros((N, 3))
    pos[1:] = radials[:, np.newaxis] * dirs
    return pos

def _cartesian_to_radial(pos):
    """Extract 54 radial coordinates from Cartesian positions."""
    dirs = _ALL_DIRS
    # radial_i = pos[i+1] · dir_i (since dirs are unit vectors)
    radials = np.sum(pos[1:] * dirs, axis=1)
    return radials

# Precompute all directions
_ALL_DIRS = _get_all_directions()

def _optimize_radii():
    """Optimize the three shell radii to minimize LJ energy."""
    def energy_from_radii(params):
        r_inner, r_edge_mid, r_vertex = params
        if r_inner < 0.3 or r_edge_mid < 0.5 or r_vertex < 0.5:
            return 1e10
        pos = _mackay_icosahedron_55(r_inner, r_edge_mid, r_vertex)
        return _lj_energy(pos)
    
    r0 = [1.054, 1.755, 2.000]
    result = minimize(energy_from_radii, r0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
    return result.x, result.fun

def _radial_energy(radials):
    """Compute LJ energy given 54 radial coordinates."""
    pos = _radial_to_cartesian(radials)
    return _lj_energy(pos)

def _radial_grad(radials):
    """Compute gradient of LJ energy w.r.t. 54 radial coordinates."""
    pos = _radial_to_cartesian(radials)
    grad_pos = _lj_energy_grad(pos)
    # grad_radial_i = grad_pos[i+1] · dir_i
    dirs = _ALL_DIRS
    grad_rad = np.sum(grad_pos[1:] * dirs, axis=1)
    return grad_rad

def _optimize_radial_manifold(pos_initial):
    """Optimize on the 54D radial manifold (fixed icosahedral directions).
    This is a lower-dimensional problem that should converge faster."""
    radials0 = _cartesian_to_radial(pos_initial)
    
    # Bound radii to reasonable range
    bounds = [(0.4, 3.5)] * 54
    
    result = minimize(
        _radial_energy,
        radials0,
        jac=_radial_grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 3000, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 50}
    )
    
    pos_opt = _radial_to_cartesian(result.x)
    return pos_opt, result.fun


def construct():
    start_time = time.time()
    
    # Step 1: Optimize shell radii (fast, ~1s)
    try:
        radii, radius_energy = _optimize_radii()
    except Exception:
        radii = [1.054, 1.755, 2.000]
        radius_energy = _lj_energy(_mackay_icosahedron_55(1.054, 1.755, 2.000))
    
    pos_mackay = _mackay_icosahedron_55(radii[0], radii[1], radii[2])
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 20:
        return pos_mackay
    
    # Step 2: Optimize on the 54D radial manifold (lower-dimensional, faster convergence)
    best_pos = pos_mackay
    best_energy = radius_energy
    
    try:
        pos_radial, e_radial = _optimize_radial_manifold(pos_mackay)
        if e_radial < best_energy:
            best_energy = e_radial
            best_pos = pos_radial
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 15:
        return best_pos
    
    # Step 3: Full 165D L-BFGS relaxation
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))
    
    def grad_func(x):
        return _lj_energy_grad(x.reshape(-1, 3)).flatten()
    
    try:
        result = minimize(
            energy_func, 
            best_pos.flatten(), 
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 50}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 10:
        return best_pos
    
    # Step 4: High-precision polish pass with tighter tolerances
    remaining = DEADLINE_SECONDS - elapsed - 5
    if remaining > 5:
        try:
            res2 = minimize(
                energy_func,
                best_pos.flatten(),
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'gtol': 1e-12, 'ftol': 1e-16, 'maxls': 100}
            )
            if res2.fun < best_energy:
                best_energy = res2.fun
                best_pos = res2.x.reshape(-1, 3)
        except Exception:
            pass
    
    # Step 5: Verify minimum separation constraint
    diffs = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        # Fallback to Mackay structure if constraint violated
        best_pos = pos_mackay
        best_energy = radius_energy
    
    return best_pos
