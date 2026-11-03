"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def _compute_radius(c):
    """
    Compute the exact maximum radius for the given centers.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i min_k (0.5 - |c_ik|) )
    """
    if c.shape[0] < 2:
        return 0.5 - np.max(np.abs(c))
    
    # Wall clearance
    wall = 0.5 - np.max(np.abs(c))
    
    # Pairwise distances
    # Using cdist is efficient for N=50
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    return min(wall, min_pair)

def _soft_min_objective(x, beta):
    """
    Objective function for L-BFGS-B.
    Minimizes the negative of the soft-min of constraints.
    Constraints:
    1. Wall: 0.5 - |c_ik|
    2. Pair: ||c_i - c_j|| / 2
    """
    c = x.reshape(N, 3)
    
    # Wall clearances
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()
    
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    d_half = d / 2.0
    # Flatten only upper triangle to avoid double counting? 
    # Actually, for soft min, summing all is fine, just doubles the weight.
    # To be consistent, let's sum all pairs i<j.
    i, j = np.triu_indices(N, k=1)
    d_half_upper = d_half[i, j]
    
    # Combine all constraint values
    # We want to maximize the minimum of these values.
    # So we minimize -min(vals).
    # Soft min: -1/beta * log(sum(exp(-beta * vals)))
    vals = np.concatenate([wall_flat, d_half_upper])
    
    # Numerical stability: shift by min
    m = np.min(vals)
    # If beta is large, exp(-beta * (vals - m)) can overflow if vals-m is negative large?
    # No, vals-m >= 0. So exp(-beta * (vals-m)) <= 1.
    # If vals-m is large positive, exp is 0.
    # So sum is at least 1.
    # log(sum) is safe.
    
    # However, if beta is very large, this becomes ill-conditioned.
    # We will ramp up beta slowly.
    
    sum_exp = np.sum(np.exp(-beta * (vals - m)))
    soft_min_val = m - np.log(sum_exp) / beta
    
    return -soft_min_val

def _grad_soft_min(x, beta):
    """
    Gradient of the soft-min objective.
    f(x) = - (m - 1/beta * log(sum(exp(-beta * (v_k - m)))))
    f(x) = -m + 1/beta * log(S) where S = sum(exp(-beta * (v_k - m)))
    
    This is complex to derive analytically with the shifting m.
    Let's use finite differences or rely on L-BFGS-B's numerical gradient?
    L-BFGS-B uses numerical gradient by default if not provided.
    For N=50, 150 variables, numerical gradient is 150 evaluations per step.
    This might be slow.
    
    Let's try to provide an analytical gradient or use a simpler objective.
    
    Alternative: Use the "log-sum-exp" form directly without the min-shift inside the log, 
    but that requires careful handling of large beta.
    
    Let's stick to numerical gradient for now but optimize the function evaluation.
    The function evaluation is the bottleneck.
    cdist is O(N^2). N=50 is small.
    
    Actually, let's try a different approach: **Penalty Method**.
    Maximize r subject to constraints.
    Minimize -r + penalty(violations).
    
    Or, simpler: **Coordinate Descent** with a "push" step.
    For each point i:
      1. Calculate gradient of min_dist w.r.t c_i.
      2. Move c_i in direction of gradient.
      
    Let's stick to the Continuation Method with L-BFGS-B and numerical gradients, 
    but ensure the function is fast.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Initialization: Farthest Point Sampling from a dense grid
    # Grid size 12x12x12 = 1728 points
    grid_size = 12
    ticks = (np.arange(grid_size) + 0.5) / grid_size - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Shuffle to break symmetry
    rng = np.random.default_rng(42)
    indices = np.arange(len(grid))
    rng.shuffle(indices)
    grid_shuffled = grid[indices]
    
    selected = [grid_shuffled[0]]
    for i in range(1, len(grid_shuffled)):
        if len(selected) >= N:
            break
        p = grid_shuffled[i]
        # Min distance to selected
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.1: # Threshold to ensure spread
            selected.append(p)
            
    # If we didn't get enough, add more
    while len(selected) < N:
        if time.time() > deadline - 10:
            break
        p = grid_shuffled[rng.integers(len(grid_shuffled))]
        # Check if not too close to existing
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if len(dists) == 0 or np.min(dists) > 0.05:
            selected.append(p)
        else:
             # Just pick a random point if we are stuck
             p = rng.uniform(-0.45, 0.45, size=3)
             selected.append(p)

    c = np.array(selected[:N])
    
    # 2. Continuation Optimization
    # We will ramp up beta from small to large.
    # This smooths the objective function, allowing the optimizer to find the basin.
    
    best_c = c.copy()
    best_val = _compute_radius(best_c)
    
    # Define the objective wrapper
    def objective(x):
        return _soft_min_objective(x, current_beta)
        
    betas = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
            
        current_beta = beta
        
        # Run L-BFGS-B
        try:
            res = minimize(objective, best_c.ravel(), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            new_c = res.x.reshape(N, 3)
            # Clip to valid region
            new_c = np.clip(new_c, -0.499, 0.499)
            
            new_val = _compute_radius(new_c)
            
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
                
        except Exception:
            pass
            
    # 3. Final Polish with high beta
    if time.time() < deadline - 10:
        current_beta = 2000
        try:
            res = minimize(objective, best_c.ravel(), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            new_c = res.x.reshape(N, 3)
            new_c = np.clip(new_c, -0.4999, 0.4999)
            new_val = _compute_radius(new_c)
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
        except Exception:
            pass
            
    # 4. Ensure valid output
    best_c = np.clip(best_c, -0.5, 0.5)
    best_c = np.asarray(best_c, dtype=float)
    
    if best_c.shape != (N, 3) or not np.all(np.isfinite(best_c)):
        # Fallback
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
        
    return best_c

# EVOLVE-BLOCK-END
