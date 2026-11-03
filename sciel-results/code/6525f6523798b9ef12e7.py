import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_score(centers):
    # centers shape (N, 3)
    # return the actual radius
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    # The score definition: min( min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|) )
    # Note: min_k (1/2 - |c_ik|) is 0.5 - max_k |c_ik|
    
    # Distances between pairs
    # To avoid O(N^2) memory if N was huge, but N=50 is small.
    # (50, 50, 3) -> (50, 50)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Boundary distances
    # max_k |c_ik|
    max_abs = np.max(np.abs(centers), axis=1)
    boundary_dists = 0.5 - max_abs
    min_boundary = np.min(boundary_dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_centers = None
    best_score = 0.0
    
    # Initial guesses
    # 1. Grid 4x4x4, random subset
    # 2. Grid 4x4x4, jittered
    
    # Generate grid points
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # grid has 64 points
    
    # Strategy 1: Random 50 from grid
    np.random.seed(42)
    idx = np.random.choice(64, 50, replace=False)
    init1 = grid[idx].copy()
    
    # Strategy 2: Jittered grid (all 64? No, we need 50)
    # Let's take the first 50 and add jitter? No, that's biased.
    # Let's take a random 50 and add jitter.
    init2 = init1 + np.random.normal(0, 0.05, (50, 3))
    # Clip to bounds
    init2 = np.clip(init2, -0.5, 0.5)
    
    # Strategy 3: More uniform random
    init3 = np.random.uniform(-0.4, 0.4, (50, 3))
    
    inits = [init1, init2, init3]
    
    # Define objective for optimizer
    # We want to maximize r.
    # Using smooth approximation.
    
    def objective(centers_flat):
        centers = centers_flat.reshape(N, 3)
        # Calculate smooth min distance
        # d_ij
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        dists = np.where(np.eye(N, dtype=bool), np.inf, dists)
        
        # Smooth min of dists
        # log-sum-exp trick for stability
        # min(d) approx -eps * log(sum(exp(-d/eps)))
        eps = 0.05
        # Scale dists to avoid overflow in exp
        # max dist is ~ sqrt(3) ~ 1.7
        # -d/eps can be large negative, exp is 0.
        # If d is small, -d/eps is close to 0.
        
        # To be safe, subtract max? No, we want min.
        # Let's just compute directly.
        # exp(-d/eps)
        # If d=0.1, eps=0.05, exp(-2) ~ 0.13
        # If d=1.0, exp(-20) ~ 2e-9
        # Sum will be dominated by small distances.
        
        d_soft = -eps * np.log(np.sum(np.exp(-dists / eps)))
        
        # Boundary
        max_abs = np.max(np.abs(centers), axis=1)
        b = 0.5 - max_abs
        b_soft = -eps * np.log(np.sum(np.exp(-b / eps)))
        
        # Combine
        # r = min(d_soft/2, b_soft)
        # smooth min of A and B: -eps * log(exp(-A/eps) + exp(-B/eps))
        A = d_soft / 2.0
        B = b_soft
        r_soft = -eps * np.log(np.exp(-A / eps) + np.exp(-B / eps))
        
        return -r_soft # Minimize negative radius
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                objective,
                init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            opt_centers = res.x.reshape(N, 3)
            score = compute_score(opt_centers)
            
            if score > best_score:
                best_score = score
                best_centers = opt_centers.copy()
                
        except Exception as e:
            pass

    # If no optimization ran or failed, use best init
    if best_centers is None:
        best_centers = init1
        best_score = compute_score(best_centers)
    else:
        # Check if best init was better than optimized one (unlikely but possible)
        for init in inits:
            s = compute_score(init)
            if s > best_score:
                best_score = s
                best_centers = init.copy()

    return best_centers
