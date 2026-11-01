"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + L-BFGS relaxation."""
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
    if n < 2:
        return 0.0
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
    np.fill_diagonal(r2, 1.0) # Avoid division by zero on diagonal
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    # Energy per pair
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    # Gradient calculation
    # dE/dr = 4 * (-12/r^13 + 6/r^7)
    # dE/dx_i = sum_j (dE/dr_ij * (x_i - x_j) / r_ij)
    # Note: factor = dE/dr * 1/r = dE/dr^2 * 2 * r^2 ... let's use standard form
    # Force on i from j: F_ij = 24 * (2/r^13 - 1/r^7) * r_vec_ij
    # Energy gradient is -Force? No, minimize energy.
    # dU/dr = 4(-12r^-13 + 6r^-7)
    # dU/dx_i = sum_{j != i} dU/dr_ij * (x_i - x_j)/r_ij
    # = sum_{j != i} 4(-12r^-13 + 6r^-7) * (x_i - x_j)/r_ij
    # = sum_{j != i} 4(-12r^-14 + 6r^-8) * (x_i - x_j)
    # Let's compute coefficient c_ij = 4 * (-12/r_ij^14 + 6/r_ij^8)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # r2 is r^2. r^-14 = r2^-7. r^-8 = r2^-4
        coeff = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = sum_j coeff_ij * (pos_i - pos_j)
    # pos_i * sum_j coeff_ij - sum_j coeff_ij * pos_j
    sum_coeff = np.sum(coeff, axis=1, keepdims=True)
    sum_coeff_pos = np.sum(coeff[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_coeff - sum_coeff_pos
    
    return energy, grad

def _get_mackay_icosahedron_positions():
    """
    Construct the ideal Mackay icosahedron geometry for 55 atoms.
    1 center
    12 first shell (vertices of icosahedron)
    30 second shell (edge midpoints)
    12 third shell (vertices of icosahedron, scaled)
    
    Radii are chosen to approximate the relaxed minimum.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit vectors
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges (pairs of vertices at minimum distance)
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    # Edge midpoints (30 atoms) - these form the second shell
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    # Shell radii. 
    # Literature values for relaxed Mackay icosahedron are approximately:
    # R1 ~ 1.05, R2 ~ 1.75, R3 ~ 1.95
    # We will use these as seeds for the full relaxation.
    r1 = 1.05
    r2 = 1.75
    r3 = 1.95
    
    pos_s1 = verts_unit * r1
    pos_s2 = edge_mids * r2
    pos_s3 = verts_unit * r3
    
    center = np.array([[0.0, 0.0, 0.0]])
    
    pos = np.vstack([center, pos_s1, pos_s2, pos_s3])
    return pos

def construct():
    """Construct 55-atom LJ cluster using ideal Mackay seed + L-BFGS relaxation."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Build ideal Mackay geometry
    pos = _get_mackay_icosahedron_positions()
    
    # 2. Relax using L-BFGS-B
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    try:
        # Use L-BFGS-B with analytical gradient
        # maxiter should be sufficient for a good seed
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-14, 'maxls': 50}
        )
        
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
            
    except Exception as e:
        print(f"Optimization error: {e}", file=sys.stderr)
        # If optimization fails, return the seed (which is already very good)
        pass
    
    # 3. Final check and return
    # Ensure no atoms are closer than 0.5 (should be satisfied by LJ minimum)
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    if min_dist_found < 0.5:
        # Scale up slightly if constraint violated
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
        # Re-relax briefly if we had to scale? 
        # Usually LJ minima are well above 0.5 (closest pair ~1.12)
        
    return best_pos
# EVOLVE-BLOCK-END
