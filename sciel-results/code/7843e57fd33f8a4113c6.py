"""Lennard-Jones 55-atom global minimum: Symmetry-constrained manifold optimization."""
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
    v_outer = v_inner.copy()
    
    return v_inner, v_edge, v_outer

V_INNER, V_EDGE, V_OUTER = _get_icosahedron_geometry()

def _rotation_matrix_from_angles(theta, phi, psi):
    """Compute rotation matrix from Euler angles (XYZ convention)."""
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    cs, ss = np.cos(psi), np.sin(psi)
    
    R = np.array([
        [cs*cp, cs*sp*st - ss*ct, cs*sp*ct + ss*st],
        [ss*cp, ss*sp*st + cs*ct, ss*sp*ct - cs*st],
        [-sp,   cp*st,            cp*ct]
    ])
    return R

def _construct_from_3radial_params(params):
    """
    Construct positions from 9 parameters: [r1, r2, r3, theta, phi, psi, tx, ty, tz]
    r1: inner shell radius (12 atoms)
    r2: edge midpoint radius (30 atoms)
    r3: outer vertex radius (12 atoms)
    theta, phi, psi: Euler rotation angles
    tx, ty, tz: translation
    """
    r1, r2, r3 = params[0], params[1], params[2]
    theta, phi, psi = params[3], params[4], params[5]
    t_vec = params[6:9]
    
    # Build base positions
    inner = V_INNER * r1
    edge = V_EDGE * r2
    outer = V_OUTER * r3
    
    scaled = np.vstack([inner, edge, outer])  # 54 unit vectors scaled
    
    # Apply rotation
    R = _rotation_matrix_from_angles(theta, phi, psi)
    rotated = scaled @ R.T
    
    # Apply translation
    translated = rotated + t_vec[np.newaxis, :]
    
    # Prepend center atom at origin
    center = np.array([[0.0, 0.0, 0.0]])
    pos = np.vstack([center, translated])
    
    return pos

def _manifold_objective_3radial(params):
    """Objective function for 3-radial manifold parameter optimization."""
    pos = _construct_from_3radial_params(params)
    
    # Check minimum distance constraint
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
    """High-precision polish pass."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 5.0:
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

def _manifold_optimize_3radial(initial_params, deadline=None, maxiter=200):
    """Optimize over the 9D manifold parameter space using Nelder-Mead."""
    start_time = time.time()
    best_energy = _manifold_objective_3radial(initial_params)
    best_params = initial_params.copy()
    
    def obj(x):
        return _manifold_objective_3radial(x)
    
    x0 = initial_params.copy()
    
    try:
        result = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x.copy()
    except Exception:
        pass
    
    return best_params, best_energy

def construct():
    """
    Construct 55-atom LJ cluster by optimizing the 9D manifold parameters
    (3 radial + 3 rotation + 3 translation) using Nelder-Mead,
    then performing full 3D relaxation.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    # Multi-start over radius combinations and small rotations
    # Typical optimal radii for LJ55 Mackay icosahedron
    radius_starts = [
        (1.04, 1.76, 1.98),
        (1.05, 1.75, 1.95),
        (1.0, 1.70, 1.90),
        (1.10, 1.80, 2.00),
        (0.95, 1.65, 1.85),
        (1.02, 1.72, 1.92),
        (1.08, 1.78, 1.98),
    ]
    
    rotation_starts = [
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.0, 0.05, 0.0],
        [0.0, 0.0, 0.05],
        [0.03, 0.03, 0.03],
        [-0.03, 0.03, 0.0],
    ]
    
    best_overall_params = None
    best_overall_energy = np.inf
    
    for r1, r2, r3 in radius_starts:
        if time.time() > deadline - 40.0:
            break
        
        for rot in rotation_starts:
            if time.time() > deadline - 40.0:
                break
            
            try:
                initial_params = np.array([r1, r2, r3] + rot + [0.0, 0.0, 0.0])
                opt_params, opt_energy = _manifold_optimize_3radial(
                    initial_params, deadline=deadline, maxiter=150
                )
                if opt_energy < best_overall_energy:
                    best_overall_energy = opt_energy
                    best_overall_params = opt_params.copy()
            except Exception:
                continue
    
    # If manifold optimization failed, use a reasonable default
    if best_overall_params is None:
        best_overall_params = np.array([1.04, 1.76, 1.98, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        best_overall_energy = _manifold_objective_3radial(best_overall_params)
    
    # Construct positions from best manifold parameters
    pos_0 = _construct_from_3radial_params(best_overall_params)
    
    # Full 3D L-BFGS relaxation for final refinement
    pos_1, e_1 = _relax_lbfgs(pos_0, deadline=deadline, maxiter=500, gtol=1e-8)
    
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
