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
    params: [r1, r2, r3]
    r1: radius for 12 inner vertices
    r2: radius for 30 edge midpoints
    r3: radius for 12 outer vertices
    """
    r1, r2, r3 = params
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = V_INNER * r1
    shell2a = V_EDGE * r2
    shell2b = V_OUTER * r3
    
    pos = np.vstack([center, shell1, shell2a, shell2b])
    return pos

def _objective(params):
    pos = _construct_from_params(params)
    # Check for minimum distance violation to penalize if needed, 
    # though for icosahedron radii ~1-2, distance is usually safe
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
    
    return best_pos, best_energy

def _polish(pos, deadline=None):
    """High-precision polish pass with staged tightening."""
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    stages = [
        {'maxiter': 3000, 'gtol': 1e-12, 'maxls': 300, 'ftol': 1e-15},
        {'maxiter': 10000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 20000, 'gtol': 1e-15, 'maxls': 1000, 'ftol': 1e-17},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 10.0:
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

def _perturb_and_relax(pos, deadline, rng, max_perturbations=5):
    """Apply symmetry-breaking perturbations and relax, keeping best result."""
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    for i in range(max_perturbations):
        if deadline is not None and time.time() > deadline - 25.0:
            break
        # Random perturbation with decreasing magnitude
        scale = 0.05 * (1.0 - 0.15 * i)
        perturbed = pos + rng.normal(0, scale, size=pos.shape)
        
        # Relax the perturbed structure
        relaxed, energy = _relax_lbfgs(perturbed, deadline=deadline, maxiter=500, gtol=1e-8)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed
    
    return best_pos, best_energy

def construct():
    """
    Construct 55-atom LJ cluster by optimizing the radial parameters
    of the Mackay icosahedron model, then performing full 3D relaxation
    with symmetry-breaking perturbations.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    rng = np.random.RandomState(42)

    # Step 1: Optimize the 3 radial parameters (r1, r2, r3)
    starts = []
    for r1 in np.linspace(1.00, 1.08, 5):
        for r2 in np.linspace(1.72, 1.80, 5):
            for r3 in np.linspace(1.94, 2.02, 5):
                starts.append([r1, r2, r3])
    starts.append([1.04, 1.76, 1.98])
    starts.append([1.05, 1.75, 1.95])
    starts.append([1.03, 1.74, 1.96])
    starts.append([1.06, 1.78, 2.00])
    starts.append([1.02, 1.72, 1.94])
    starts.append([1.04, 1.76, 2.02])
    starts.append([1.04, 1.72, 1.98])
    starts.append([1.04, 1.80, 1.98])
    
    best_params = None
    best_param_energy = float('inf')
    
    for start in starts:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(
                _objective, 
                start, 
                method='Nelder-Mead', 
                options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_param_energy:
                best_param_energy = res.fun
                best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        best_params = np.array([1.04, 1.76, 1.98])
        best_param_energy = _objective(best_params)
        
    # Construct the initial structure from optimized parameters
    pos_0 = _construct_from_params(best_params)
    
    # Step 2: Full 3D L-BFGS relaxation
    pos_1, e_1 = _relax_lbfgs(pos_0, deadline=deadline, maxiter=1000, gtol=1e-8)
    
    # Step 3: Symmetry-breaking perturbations to explore nearby minima
    pos_2, e_2 = _perturb_and_relax(pos_1, deadline, rng, max_perturbations=5)
    
    if e_2 < e_1:
        pos_1 = pos_2
        e_1 = e_2
    
    # Step 4: High precision polish
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
