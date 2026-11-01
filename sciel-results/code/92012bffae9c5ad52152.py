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
        edges: list of 30 tuples (i, j) representing edges between inner vertices.
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
    
    return v_inner, edges

V_INNER, EDGES = _get_icosahedron_geometry()

def _construct_from_params(params):
    """
    params: [r1, r2, t_1, ..., t_30]
    r1: radius for 12 inner vertices
    r2: base radius scale for outer shell (midpoint radius)
    t_k: edge fraction for each of the 30 edges, in [0, 1]
         t=0.5 corresponds to the midpoint (standard Mackay)
         The outer atom on edge k is at:
             pos = (1 - t_k) * v_inner[i] + t_k * v_inner[j]
         scaled by r2 / ||midpoint_unit_vector|| * ||v_inner[i]|| ... 
         Actually, let's be more precise.
         
         In the standard Mackay icosahedron, the outer shell atoms are at the 
         midpoints of the edges of the inner icosahedron, scaled by a factor.
         
         Let's define the outer atom position on edge (i, j) as:
             p_outer = r2 * normalize( (1 - t) * v_inner[i] + t * v_inner[j] )
         
         Wait, the standard Mackay icosahedron has the outer atoms at the 
         midpoints of the edges of the inner icosahedron, but at a different radius.
         
         Let's use a simpler parameterization:
         The outer atom on edge (i, j) with parameter t is:
             p_outer = r2 * normalize( (1 - t) * v_inner[i] + t * v_inner[j] )
         
         This ensures the atom lies on the line segment between v_inner[i] and v_inner[j],
         but scaled to have norm r2. When t=0.5, this is the midpoint direction scaled by r2.
    """
    r1 = params[0]
    r2 = params[1]
    t_vals = params[2:]
    
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = V_INNER * r1
    
    outer_atoms = []
    for k, (i, j) in enumerate(EDGES):
        t = t_vals[k]
        # Clamp t to [0, 1] to ensure the point lies on the edge segment
        t = np.clip(t, 0.0, 1.0)
        # Point on the edge in the direction of the unit vectors
        point_on_edge = (1.0 - t) * V_INNER[i] + t * V_INNER[j]
        norm = np.linalg.norm(point_on_edge)
        if norm < 1e-10:
            # Fallback to midpoint if degenerate
            point_on_edge = (V_INNER[i] + V_INNER[j]) / 2.0
            norm = np.linalg.norm(point_on_edge)
        # Scale to radius r2
        outer_atom = r2 * point_on_edge / norm
        outer_atoms.append(outer_atom)
    
    shell2 = np.array(outer_atoms)
    
    pos = np.vstack([center, shell1, shell2])
    return pos

def _objective(params):
    pos = _construct_from_params(params)
    # Check for minimum distance violation
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 0.5:
        return 1e10
    return _lj_energy(pos)

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
    
    stages = [
        {'maxiter': 2000, 'gtol': 1e-10, 'maxls': 300, 'ftol': 1e-14},
        {'maxiter': 5000, 'gtol': 1e-12, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-14, 'maxls': 1000, 'ftol': 1e-18},
        {'maxiter': 20000, 'gtol': 1e-16, 'maxls': 2000, 'ftol': 1e-20},
        {'maxiter': 50000, 'gtol': 1e-15, 'maxls': 5000, 'ftol': 1e-20},
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
    Construct 55-atom LJ cluster by optimizing the radial and edge-fraction parameters
    of the Mackay icosahedron model, then performing full 3D relaxation.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    # Step 1: Optimize the parameters (r1, r2, t_1, ..., t_30)
    # The theoretical Mackay icosahedron has t_k = 0.5 for all k (midpoints).
    # Typical optimal radii for LJ55 are around r1 ~ 1.04, r2 ~ 1.76.
    
    # Initial guess: all t = 0.5 (standard Mackay)
    t_init = np.full(30, 0.5)
    
    # Multi-start optimization on the parameter space
    # We'll use a few different starting points for r1, r2 and small perturbations of t
    
    starts = []
    for r1 in [1.02, 1.04, 1.06]:
        for r2 in [1.74, 1.76, 1.78]:
            # Start with all midpoints
            starts.append([r1, r2] + list(t_init))
            # Add small random perturbations to t
            for _ in range(3):
                t_perturbed = t_init + np.random.normal(0, 0.05, 30)
                t_perturbed = np.clip(t_perturbed, 0.1, 0.9)
                starts.append([r1, r2] + list(t_perturbed))
    
    best_params = None
    best_param_energy = float('inf')
    
    for start in starts:
        if time.time() > deadline - 15.0:
            break
        try:
            res = minimize(
                _objective, 
                start, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_param_energy:
                best_param_energy = res.fun
                best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        # Fallback to standard Mackay with typical radii
        best_params = np.array([1.04, 1.76] + list(t_init))
        best_param_energy = _objective(best_params)
        
    # Construct the initial structure from optimized parameters
    pos_0 = _construct_from_params(best_params)
    
    # Step 2: Full 3D L-BFGS relaxation
    pos_1, e_1 = _relax_lbfgs(pos_0, deadline=deadline, maxiter=1000, gtol=1e-8)
    
    # Intermediate refinement
    if deadline is not None and time.time() < deadline - 30.0:
        pos_2, e_2 = _relax_lbfgs(pos_1, deadline=deadline, maxiter=2000, gtol=1e-10)
        if e_2 < e_1:
            pos_1 = pos_2
    
    # High precision polish
    pos_final, e_final = _polish(pos_1, deadline=deadline)
    
    # Verify constraints
    diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        pos_final = pos_final * scale
        pos_final, e_final = _relax_lbfgs(pos_final, deadline=deadline, maxiter=200, gtol=1e-10)

    return pos_final
# EVOLVE-BLOCK-END
