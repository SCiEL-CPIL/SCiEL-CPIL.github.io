"""Lennard-Jones 55-atom global minimum: Parameterized Mackay icosahedron optimization."""
import numpy as np
import time
import sys

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_icosahedron_geometry():
    """
    Returns:
        v_inner: 12 unit vectors for the first shell (vertices).
        v_edge: 30 unit vectors for the second shell (edge midpoints).
        v_outer: 12 unit vectors for the second shell (vertices, same directions as inner).
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    v_inner = verts / norms
    
    # Find edges
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(v_inner[i] - v_inner[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    v_edge = []
    for i, j in edges:
        mid = (v_inner[i] + v_inner[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            v_edge.append(mid / norm)
    v_edge = np.array(v_edge)
    
    # Outer vertices have same directions as inner vertices in the standard Mackay model
    # In a relaxed structure, they might rotate slightly, but radial parameterization assumes fixed angular directions
    v_outer = v_inner.copy()
    
    return v_inner, v_edge, v_outer

V_INNER, V_EDGE, V_OUTER = _get_icosahedron_geometry()

def _construct_from_params(params):
    """
    params: array of 55 radial distances
    params[0]: radius for center atom (should be 0)
    params[1:13]: radii for 12 inner shell vertices
    params[13:43]: radii for 30 edge midpoint positions
    params[43:55]: radii for 12 outer shell vertices
    """
    n = len(params)
    pos = np.zeros((n, 3))
    
    # Center atom stays at origin
    # pos[0] = [0, 0, 0]
    
    # Inner shell: 12 atoms
    for i in range(12):
        r = params[1 + i]
        pos[1 + i] = V_INNER[i] * r
    
    # Edge midpoints: 30 atoms
    for i in range(30):
        r = params[13 + i]
        pos[13 + i] = V_EDGE[i] * r
    
    # Outer shell: 12 atoms
    for i in range(12):
        r = params[43 + i]
        pos[43 + i] = V_OUTER[i] * r
    
    return pos

def _radial_objective(params):
    """Objective function for radial manifold optimization."""
    pos = _construct_from_params(params)
    # Check for minimum distance violation
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 0.5:
        return 1e10
    return _lj_energy(pos)

def _radial_grad(params):
    """Compute gradient of energy w.r.t. radial parameters using finite differences."""
    n = len(params)
    grad = np.zeros(n)
    h = 1e-6
    f0 = _radial_objective(params)
    
    for i in range(n):
        params_plus = params.copy()
        params_plus[i] += h
        f_plus = _radial_objective(params_plus)
        grad[i] = (f_plus - f0) / h
    
    return grad

def _optimize_radial_manifold(initial_params, deadline, maxiter=5000):
    """Optimize over the 55-dimensional radial manifold."""
    start_time = time.time()
    best_energy = _radial_objective(initial_params)
    best_params = initial_params.copy()
    
    def obj_and_grad(x):
        e = _radial_objective(x)
        g = _radial_grad(x)
        return e, g
    
    try:
        result = minimize(
            obj_and_grad, initial_params, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x
    except Exception:
        pass
    
    return best_params, best_energy

def _relax_lbfgs(pos, deadline=None, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    if deadline is not None and time.time() > deadline - 5.0:
        return best_pos, best_energy
    
    return best_pos, best_energy

def _polish(pos, deadline=None):
    """High-precision polish pass with more aggressive stages."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    # More aggressive polishing with tighter tolerances and more iterations
    stages = [
        {'maxiter': 2000, 'gtol': 1e-10, 'maxls': 300, 'ftol': 1e-14},
        {'maxiter': 5000, 'gtol': 1e-12, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-14, 'maxls': 1000, 'ftol': 1e-18},
        {'maxiter': 20000, 'gtol': 1e-16, 'maxls': 2000, 'ftol': 1e-20},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 8.0:
            break
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos, best_energy

def construct():
    """
    Construct 55-atom LJ cluster by optimizing the radial parameters
    of the Mackay icosahedron model, then performing full 3D relaxation.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    # Step 1: Optimize the 55 radial parameters (one per atom)
    # This allows each atom to find its optimal radial distance
    # while maintaining the icosahedral angular framework
    
    # Initial guess: uniform radii per shell
    # r1 ~ 1.04 for inner shell, r2 ~ 1.76 for edge midpoints, r3 ~ 1.98 for outer shell
    initial_params = np.zeros(55)
    initial_params[0] = 0.0  # center
    initial_params[1:13] = 1.04   # inner shell
    initial_params[13:43] = 1.76  # edge midpoints
    initial_params[43:55] = 1.98  # outer shell
    
    # Optimize over the radial manifold
    best_params, best_param_energy = _optimize_radial_manifold(initial_params, deadline, maxiter=3000)
    
    # Construct the initial structure from optimized parameters
    pos_0 = _construct_from_params(best_params)
    
    # Step 2: Full 3D L-BFGS relaxation
    # This allows for any slight symmetry breaking or distortions
    # that the radial-only model cannot capture
    
    # Initial coarse relaxation
    pos_1, e_1 = _relax_lbfgs(pos_0, deadline=deadline, maxiter=500, gtol=1e-8)
    
    # Intermediate refinement if time permits
    if time.time() < deadline - 40.0:
        pos_2, e_2 = _relax_lbfgs(pos_1, deadline=deadline, maxiter=1000, gtol=1e-10)
        if e_2 < e_1:
            pos_1 = pos_2
            e_1 = e_2
    
    # High precision polish
    pos_final, e_final = _polish(pos_1, deadline=deadline)
    
    # Verify constraints
    diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        # This should not happen for the optimized radii, but just in case
        scale = 0.5 / min_dist_found * 1.01
        pos_final = pos_final * scale
        # Re-relax if scaled
        pos_final, e_final = _relax_lbfgs(pos_final, deadline=deadline, maxiter=200, gtol=1e-10)

    return pos_final
# EVOLVE-BLOCK-END
