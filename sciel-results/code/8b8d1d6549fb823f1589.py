"""Rewritten program using direct FCC lattice construction and SciPy optimization."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def _lj_energy_and_grad(pos):
    """
    Compute Lennard-Jones energy and gradient for a cluster of N atoms.
    pos: (N, 3) array of coordinates.
    Returns: energy (float), grad (N, 3) array.
    """
    N = len(pos)
    # Pairwise differences
    # diff[i,j] = pos[i] - pos[j]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    
    # Squared distances
    # dist2[i,j] = |diff[i,j]|^2
    dist2 = np.sum(diff**2, axis=2)
    
    # Mask diagonal (self-interaction)
    np.fill_diagonal(dist2, 1.0) # Avoid division by zero
    
    # Inverse distances
    inv_r2 = 1.0 / dist2
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    
    # Energy term: 4 * (inv_r12 - inv_r6)
    # We need to sum over all unique pairs, so we sum all and divide by 2
    # Note: The diagonal is 0 in energy calculation, so filling with 1.0 makes term 0
    # 4*(1 - 1) = 0.
    energy = 2.0 * np.sum(4.0 * (inv_r12 - inv_r6))
    
    # Gradient
    # dE/dr = 4 * (-12*r^-13 + 6*r^-7) = 24*r^-7 * (1 - 2*r^-6)
    # dE/dpos_i = sum_{j != i} dE/dr_ij * (r_i - r_j) / r_ij
    # Let u = r^-6. dE/dr = 24 * r^-7 * (1 - 2u)
    # dE/dpos_i = sum_j 24 * r^-7 * (1 - 2u) * (pos_i - pos_j) / r
    #            = sum_j 24 * r^-6 * (1 - 2u) * (pos_i - pos_j) / r^2
    #            = sum_j 24 * u * (1 - 2u) * (pos_i - pos_j) / r^2
    #            = sum_j 24 * u * (1 - 2u) * (pos_i - pos_j) * inv_r2
    
    # Factor for each pair
    # factor[i,j] = 24 * inv_r6[i,j] * (1 - 2*inv_r6[i,j]) * inv_r2[i,j]
    # Note: We sum over j for each i.
    # grad[i] = sum_j factor[i,j] * (pos[i] - pos[j])
    
    # Wait, let's re-derive carefully.
    # E = sum_{i<j} 4 (r_ij^-12 - r_ij^-6)
    # dE/dx_i = sum_{j != i} 4 (-12 r_ij^-13 + 6 r_ij^-7) * (x_i - x_j) / r_ij
    #         = sum_{j != i} 24 r_ij^-7 (1 - 2 r_ij^-6) * (x_i - x_j) / r_ij
    #         = sum_{j != i} 24 r_ij^-6 (1 - 2 r_ij^-6) * (x_i - x_j) / r_ij^2
    
    # Let A[i,j] = 24 * inv_r6[i,j] * (1 - 2*inv_r6[i,j]) * inv_r2[i,j]
    # Then grad[i] = sum_j A[i,j] * (pos[i] - pos[j])
    
    # We can compute this efficiently.
    # A is (N, N).
    # grad = A @ (pos[i] - pos[j])? No, element-wise multiplication.
    # grad[i] = sum_j A[i,j] * pos[i] - sum_j A[i,j] * pos[j]
    # grad[i] = pos[i] * sum_j A[i,j] - sum_j A[i,j] * pos[j]
    
    A = 24.0 * inv_r6 * (1.0 - 2.0 * inv_r6) * inv_r2
    # Set diagonal to 0
    np.fill_diagonal(A, 0.0)
    
    row_sums = np.sum(A, axis=1)
    # grad[i] = row_sums[i] * pos[i] - (A @ pos)[i]
    grad = row_sums[:, np.newaxis] * pos - (A @ pos)
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe (150s total, leave margin)
    
    # The Lennard-Jones minimum distance is 2^(1/6)
    r_min = 2.0 ** (1.0 / 6.0)
    
    # Construct FCC lattice
    scale = r_min / np.sqrt(2.0)
    max_coord = 5
    
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append((i, j, k))
    
    pts = np.array(points, dtype=float)
    pts *= scale
    
    dists = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists)
    cluster = pts[sorted_indices[:N]].copy()
    
    # Relax using SciPy L-BFGS-B
    try:
        # Use L-BFGS-B as it handles bounds well and is robust
        # No bounds specified, but L-BFGS-B is generally good
        res = minimize(_lj_energy_and_grad, cluster, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000})
        
        if res.success or res.fun < _lj_energy_and_grad(cluster)[0]:
             cluster = res.x
             
    except Exception as e:
        pass
        
    if time.time() > deadline:
        # If we are over time, we just return what we have. 
        # The minimize call usually finishes quickly for this size.
        pass

    return cluster
# EVOLVE-BLOCK-END
