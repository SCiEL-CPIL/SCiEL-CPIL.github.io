import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    c = centers
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    
    # Boundary clearance
    # min_i min_k (0.5 - |c_ik|)
    # This is 0.5 - max_i max_k |c_ik|
    min_boundary = 0.5 - np.max(np.abs(c))
    
    r = min(min_pairwise / 2, min_boundary)
    return r

def objective(centers_flat):
    """
    Minimize the sum of inverse squared distances to push points apart.
    This approximates maximizing the minimum distance.
    """
    c = centers_flat.reshape(N, 2)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1e10) # Avoid self-interaction
    
    # Sum of inverse squares
    # Add small epsilon to avoid division by zero if points coincide
    inv_d2 = 1.0 / (dists_sq + 1e-6)
    pair_energy = np.sum(inv_d2)
    
    # Boundary distances
    # Distance to each boundary: 0.5 - |x|, 0.5 - |y|
    dist_x = 0.5 - np.abs(c[:, 0])
    dist_y = 0.5 - np.abs(c[:, 1])
    
    # If a point is outside, dist is negative. We want to penalize this heavily.
    # Using inverse square on negative numbers is problematic.
    # Better: penalize violation.
    # Or simply: if any dist < 0, return large penalty.
    if np.any(dist_x < 0) or np.any(dist_y < 0):
        return 1e10
    
    # Sum of inverse squares of boundary distances
    # Add epsilon
    b_energy = np.sum(1.0 / (dist_x**2 + 1e-6)) + np.sum(1.0 / (dist_y**2 + 1e-6))
    
    return pair_energy + b_energy

def construct():
    deadline = time.time() + 140
    np.random.seed(42)
    
    best_centers = None
    best_val = 0
    
    # Initial configurations
    inits = []
    
    # 1. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    c_grid = grid[:N].copy()
    # Add small jitter to break symmetry
    c_grid += np.random.normal(0, 0.01, size=c_grid.shape)
    c_grid = np.clip(c_grid, -0.499, 0.499)
    inits.append(c_grid)
    
    # 2. Hexagonal-like
    # Approximate hexagonal packing
    # For N=50, we can try to fit rows of 7, 8, 7, 8... or similar
    # Let's just use a distorted grid or random as other starts
    c_hex = np.random.uniform(-0.49, 0.49, (N, 2))
    inits.append(c_hex)
    
    # 3. Random
    c_rand = np.random.uniform(-0.49, 0.49, (N, 2))
    inits.append(c_rand)
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                objective,
                init.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 100000, 'ftol': 1e-10, 'gtol': 1e-8},
                bounds=[(-0.5, 0.5)] * (2 * N)
            )
            
            c_opt = result.x.reshape(N, 2)
            c_opt = np.clip(c_opt, -0.5 + 1e-8, 0.5 - 1e-8)
            
            r_opt = compute_radius(c_opt)
            
            if r_opt > best_val:
                best_val = r_opt
                best_centers = c_opt.copy()
                
        except Exception:
            pass
            
    if best_centers is None:
        # Fallback to grid
        best_centers = inits[0]
        
    return best_centers
