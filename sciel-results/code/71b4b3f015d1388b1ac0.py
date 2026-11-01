"""Lennard-Jones 55-atom global minimum: High-precision Mackay icosahedron optimization."""
import numpy as np
import time
import sys
from scipy.optimize import minimize

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
        # dU/dr = 4 * (-12 r^-13 + 6 r^-7)
        # dU/dx_i = dU/dr * dr/dx_i = dU/dr * (x_i - x_j)/r
        # r * dU/dr = 4 * (-12 r^-12 + 6 r^-6)
        # factor = 4 * (-12/r12 + 6/r6)
        # grad_i = sum_j factor_ij * (pos_i - pos_j)
        # factor_ij = 4 * (-12/r12_ij + 6/r6_ij)
        
        # Let's re-derive carefully.
        # U = sum 4(r^-12 - r^-6)
        # dU/dx_i = sum_{j!=i} 4 * (-12 r^-13 * dr/dx_i + 6 r^-7 * dr/dx_i)
        # dr/dx_i = (x_i - x_j) / r
        # dU/dx_i = sum_{j!=i} 4 * (-12 r^-12 + 6 r^-6) * (x_i - x_j)
        
        # Let A_ij = 4 * (-12/r12_ij + 6/r6_ij)
        # Then grad_i = sum_{j!=i} A_ij * (pos_i - pos_j)
        
        # A_ij is symmetric.
        # grad_i = pos_i * sum_j A_ij - sum_j A_ij * pos_j
        
        factor = 4.0 * (-12.0 / r12 + 6.0 / r6)
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_mackay_icosahedron_coords():
    """
    Returns coordinates for the Mackay icosahedron with 55 atoms.
    Shell 1: 12 atoms at radius r1
    Shell 2: 30 atoms at radius r2 (edge midpoints)
    Shell 3: 12 atoms at radius r3 (vertex extensions)
    Center: 1 atom at origin
    
    Radii are initialized close to the known minimum.
    """
    # Known optimal radii for LJ55 Mackay Icosahedron (approximate)
    # r1 ~ 1.05, r2 ~ 1.80, r3 ~ 2.05
    # Let's use slightly perturbed values to avoid symmetry traps if any, 
    # but actually the symmetric structure IS the minimum.
    r1 = 1.05
    r2 = 1.80
    r3 = 2.05
    
    # Icosahedron vertices (unit vectors)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    v_inner = verts / norms
    
    # Find edges (shortest distances between vertices)
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
    
    # Edge midpoints (normalized)
    v_edge = []
    for i, j in edges:
        mid = (v_inner[i] + v_inner[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            v_edge.append(mid / norm)
    v_edge = np.array(v_edge)
    
    # Outer vertices (same directions as inner)
    v_outer = v_inner.copy()
    
    # Construct positions
    pos = np.zeros((55, 3))
    
    # Center
    pos[0] = [0.0, 0.0, 0.0]
    
    # Inner shell
    pos[1:13] = v_inner * r1
    
    # Edge shell
    pos[13:43] = v_edge * r2
    
    # Outer shell
    pos[43:55] = v_outer * r3
    
    return pos

def _relax_lbfgs(pos, deadline=None, maxiter=1000, gtol=1e-10):
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

def _high_precision_polish(pos, deadline=None):
    """
    Perform multiple stages of L-BFGS-B with progressively tighter tolerances
    to reach the machine-precision minimum.
    """
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    # Stages with tighter tolerances
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-18},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-20},
        {'maxiter': 20000, 'gtol': 1e-18, 'maxls': 2000, 'ftol': 1e-22},
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

def construct():
    """
    Construct 55-atom LJ cluster by initializing with Mackay icosahedron
    and performing high-precision relaxation.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: evaluator kills at 150s

    # Initialize with Mackay icosahedron
    pos_0 = _get_mackay_icosahedron_coords()
    
    # Initial relaxation to settle into the local minimum basin
    pos_1, e_1 = _relax_lbfgs(pos_0, deadline=deadline, maxiter=500, gtol=1e-8)
    
    # High precision polish to reach the exact minimum
    pos_final, e_final = _high_precision_polish(pos_1, deadline=deadline)
    
    # Verify constraints: atoms must be at least 0.5 apart
    diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        # This should not happen for the Mackay icosahedron, but handle it just in case
        scale = 0.5 / min_dist_found * 1.01
        pos_final = pos_final * scale
        pos_final, e_final = _relax_lbfgs(pos_final, deadline=deadline, maxiter=200, gtol=1e-10)
        pos_final, e_final = _high_precision_polish(pos_final, deadline=deadline)

    return pos_final
# EVOLVE-BLOCK-END
