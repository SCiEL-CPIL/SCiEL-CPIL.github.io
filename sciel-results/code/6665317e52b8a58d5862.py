import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 38

def _lj_energy_and_grad(pos):
    """Calculate LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    # Avoid division by zero
    r2[np.arange(n)[:, None] == np.arange(n)[None, :]] = 1.0
    
    r6 = r2**3
    r12 = r6**2
    
    # Energy: 4 * sum (r^-12 - r^-6)
    # Note: sum over i<j. The matrix sum is 2 * sum_{i<j}.
    # So energy = 2 * 4 * sum_{i<j} (...) = 8 * sum_{i<j} ...
    # Or just sum the whole matrix and divide by 2?
    # E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # Matrix M has M_ij = r_ij^-12 - r_ij^-6.
    # sum_{i<j} M_ij = (sum_{i,j} M_ij - sum_{i} M_ii) / 2
    # M_ii = 0 (if we set r2=1, it's 1-1=0).
    # So E = 4 * (sum(M) / 2) = 2 * sum(M).
    
    pair_term = r12 - r6
    energy = 2.0 * np.sum(pair_term)
    
    # Gradient
    # dE/dr_ij = 4 * (-12 r^-13 + 6 r^-7) = 24 * (r^-7 - 2 r^-13)
    # dE/dx_i = sum_{j != i} dE/dr_ij * d r_ij / d x_i
    # r_ij = |x_i - x_j|
    # d r_ij / d x_i = (x_i - x_j) / r_ij
    # So dE/dx_i = sum_{j != i} 24 * (r_ij^-7 - 2 r_ij^-13) * (x_i - x_j) / r_ij
    # = 24 * sum_{j != i} (r_ij^-8 - 2 r_ij^-14) * (x_i - x_j)
    
    # r^-8 = (r2^-1/2)^-8 = r2^-4
    # r^-14 = r2^-7
    
    grad_factor = 24.0 * (r2**(-4.0) - 2.0 * r2**(-7.0))
    # grad_factor is a matrix G where G_ij = 24 * (r_ij^-8 - 2 r_ij^-14)
    # grad_i = sum_j G_ij * (x_i - x_j)
    # grad_i = sum_j G_ij * x_i - sum_j G_ij * x_j
    # grad_i = x_i * sum_j G_ij - (G @ x)_i
    
    sum_G = np.sum(grad_factor, axis=1)
    Gx = grad_factor @ pos
    
    grad = pos * sum_G[:, np.newaxis] - Gx
    
    return energy, grad

def _relax(positions):
    """Relax positions using L-BFGS-B."""
    # Flatten positions
    x0 = positions.flatten()
    
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos)
        return e, g
    
    # Use L-BFGS-B
    # maxiter should be reasonable
    res = minimize(fun, x0, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if res.success:
        return res.x.reshape(-1, 3)
    else:
        return positions

def _generate_fcc_cluster(n_atoms):
    """Generate n_atoms closest FCC lattice points to origin."""
    # Generate points in a box
    # Need enough points
    max_coord = 5
    points = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                if (x + y + z) % 2 == 0:
                    points.append((x, y, z))
    
    # Sort by distance squared
    points.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    
    # Take first n_atoms
    pts = np.array(points[:n_atoms], dtype=float)
    
    # Scale
    # Nearest neighbor in this grid is sqrt(2)
    # Target nearest neighbor is 2^(1/6)
    scale = 2**(1/6) / np.sqrt(2)
    pts = pts * scale
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Generate initial guess
    pos = _generate_fcc_cluster(N)
    
    # 2. Relax
    # Check time
    if time.time() < deadline - 5.0:
        pos_relaxed = _relax(pos)
        if time.time() < deadline:
            pos = pos_relaxed
    
    # 3. Optional: Multiple restarts if time permits
    # The FCC start is very strong, so one relaxation is likely sufficient.
    # However, to be robust against local minima if the FCC subset is slightly off,
    # we could try a few random perturbations.
    
    # Let's check if we have time for a second, slightly perturbed run
    if time.time() < deadline - 15.0:
        # Add small noise
        np.random.seed(42)
        pos_perturbed = pos + np.random.normal(0, 0.01, pos.shape)
        pos_perturbed_relaxed = _relax(pos_perturbed)
        
        # Compare energies
        e1, _ = _lj_energy_and_grad(pos)
        e2, _ = _lj_energy_and_grad(pos_perturbed_relaxed)
        
        if e2 < e1:
            pos = pos_perturbed_relaxed
            
    return pos
# EVOLVE-BLOCK-END
