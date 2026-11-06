"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100
TIME_LIMIT = 115.0  # Safety margin before 120s/150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_radius(centres):
    """
    Compute the maximum common radius for given centres.
    centres: (N, 3) array
    Returns: float, the maximum radius r.
    """
    n = len(centres)
    if n < 2:
        return np.min(0.5 - np.abs(centres))
    
    # Pairwise distances
    # Using broadcasting for vectorization
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(centres)
    min_boundary_dist = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _relax_initial(centres, iterations=50, step=0.01):
    """
    Simple force-directed relaxation to spread points out.
    Helps avoid local minima caused by bad initial overlaps.
    """
    centres = centres.copy()
    n = len(centres)
    
    for _ in range(iterations):
        # Compute forces
        forces = np.zeros_like(centres)
        
        # Repulsion between spheres
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0) # Avoid division by zero
        
        # We want to maximize min distance. 
        # Simple heuristic: push apart if closer than some threshold.
        # Let's use a soft potential.
        # Force magnitude proportional to 1/d^2 for d < d_target
        d_target = 0.15 # Rough estimate of desired spacing
        mask = dists < d_target
        # Invert mask for self
        mask = mask & (np.arange(n)[:, None] != np.arange(n)[None, :])
        
        # Calculate repulsive force
        # F = k * (1/d^2) * direction
        # Only apply if d < d_target
        inv_d2 = np.zeros_like(dists)
        inv_d2[mask] = 1.0 / (dists[mask] ** 2)
        
        # Force vector: diff * inv_d2
        # Sum over neighbors
        forces += np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        # Attraction to center to keep them in bounds? 
        # Or just boundary penalty.
        # Let's add a gentle pull towards center if they are near boundary
        boundary_dist = 0.5 - np.abs(centres)
        near_boundary = boundary_dist < 0.05
        # Pull inwards
        pull = np.zeros_like(centres)
        for i in range(n):
            for k in range(3):
                if near_boundary[i, k]:
                    # Pull towards 0
                    pull[i, k] = -centres[i, k] * 0.5
        
        forces += pull
        
        # Update positions
        centres += step * forces
        
        # Clip to cube
        centres = np.clip(centres, -0.49, 0.49)
        
    return centres

def _optimize_packing(centres, max_time=5.0):
    """
    Optimize the packing using Nelder-Mead.
    """
    start_time = time.time()
    n = len(centres)
    centres = centres.copy().astype(float)
    
    def objective(x):
        c = x.reshape(n, 3)
        # Ensure validity during optimization (soft penalty for going out of bounds)
        # But the main goal is to maximize min distance.
        # We can just compute the radius and return negative.
        # If points go out of bounds, the boundary distance becomes negative, 
        # so the radius becomes negative, which is bad.
        r = _compute_radius(c)
        return -r

    best_c = centres.copy()
    best_r = _compute_radius(best_c)
    
    # Nelder-Mead parameters
    # Initial simplex size
    x0 = best_c.flatten()
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        
        if time.time() - start_time > max_time:
             return best_c, best_r
             
        new_c = res.x.reshape(n, 3)
        new_r = _compute_radius(new_c)
        
        if new_r > best_r:
            best_c = new_c
            best_r = new_r
    except Exception:
        pass
        
    return best_c, best_r

def _generate_initializations():
    """
    Generate a list of initial configurations.
    """
    inits = []
    
    # 1. Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N].copy())
    
    # 2. Random with relaxation
    rng = np.random.RandomState(42)
    for seed in [0, 1, 2, 3, 42, 100, 200, 300]:
        rng = np.random.RandomState(seed)
        c = rng.uniform(-0.45, 0.45, size=(N, 3))
        # Relax to spread out
        c_relaxed = _relax_initial(c, iterations=20, step=0.02)
        inits.append(c_relaxed)
        
    return inits

def construct():
    start_time = time.time()
    best_centres = None
    best_radius = -np.inf
    
    # Get initializations
    inits = _generate_initializations()
    
    # Iterate through initializations
    for i, init_centres in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10:
            break
            
        # Quick check
        r_init = _compute_radius(init_centres)
        
        # Optimize
        # Allocate remaining time minus safety margin
        time_budget = min(TIME_LIMIT - elapsed - 5, 10.0) # Cap per iteration to ensure we try multiple
        if time_budget < 1.0:
            break
            
        opt_centres, opt_radius = _optimize_packing(init_centres, max_time=time_budget)
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centres = opt_centres
            
        # Check time
        if time.time() - start_time > TIME_LIMIT - 15:
            break
            
    # Final polish on the best solution
    if best_centres is not None:
        elapsed = time.time() - start_time
        if elapsed < TIME_LIMIT - 2:
            # Use remaining time for a final high-precision polish
            polish_time = TIME_LIMIT - elapsed - 1
            final_centres, final_radius = _optimize_packing(best_centres, max_time=polish_time)
            if final_radius > best_radius:
                best_centres = final_centres
                best_radius = final_radius
                
    # Ensure validity
    if best_centres is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centres = grid[:N].copy()
        
    best_centres = np.clip(best_centres, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centres
# EVOLVE-BLOCK-END
