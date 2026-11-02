import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200

def compute_radius(P):
    """
    Compute the maximum radius for a given set of centers P.
    P: (N, 2) array of centers.
    Returns the radius r.
    """
    # Pairwise distances
    # cdist is optimized
    dists = cdist(P, P)
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    # P is in [-0.5, 0.5]
    # Distance to edge is min(0.5 - x, x + 0.5) for x in [-0.5, 0.5]
    # Which is 0.5 - |x|
    bound_dists = 0.5 - np.abs(P)
    min_bound_dist = np.min(bound_dists)
    
    return min(min_pair_dist, min_bound_dist)

def soft_min_objective(P, beta=1.0):
    """
    Differentiable objective for optimization.
    Minimizes -softmin(distances).
    """
    dists = cdist(P, P)
    np.fill_diagonal(dists, np.inf)
    
    bound_dists = 0.5 - np.abs(P)
    
    # Combine all distances into one array
    # To avoid huge memory, we can just compute the log-sum-exp on the flattened array
    # But we need to be careful with the inf on diagonal.
    
    # Flatten pairwise dists (excluding diagonal)
    # Mask for upper triangle to avoid double counting? 
    # Actually, for softmin, double counting doesn't change the argmin, 
    # but it scales the result slightly. 
    # It's better to use unique pairs.
    
    # For simplicity and speed in the polish phase, 
    # we can just use all pairwise distances and boundary distances.
    # N=200, pairs ~ 20k. This is fine.
    
    all_dists = np.concatenate([dists.ravel(), bound_dists.ravel()])
    
    # Soft min: -log(sum(exp(-d/beta)))
    # To avoid overflow/underflow, use log-sum-exp trick
    # max_d = np.max(all_dists)
    # soft_min = max_d + beta * np.log(np.sum(np.exp(-(all_dists - max_d)/beta)))
    # Wait, softmin(x) = -logsumexp(-x/beta) * beta
    
    # Let's just use a simple approximation or standard logsumexp
    # scipy.special.logsumexp is available
    from scipy.special import logsumexp
    
    # We want to minimize the radius, so we maximize the "soft min".
    # Objective to minimize: -soft_min
    # soft_min = -beta * logsumexp(-all_dists / beta)
    
    obj = beta * logsumexp(-all_dists / beta)
    return obj

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    # 1. Initialization: Hexagonal Grid
    # Generate a hex grid that covers the square
    # Spacing s
    s = 0.075 
    # Rows
    # y = k * s * sqrt(3)/2
    # x = i * s + (k % 2) * s/2
    
    points = []
    y = 0.0
    k = 0
    while y <= 0.5:
        x_start = -0.5 + (s/2 if k % 2 == 1 else 0)
        x = x_start
        while x <= 0.5:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append((x, y))
            x += s
        y += s * np.sqrt(3) / 2
        k += 1
    
    P = np.array(points)
    
    # If we have too many points, randomly subsample
    if len(P) > N:
        indices = np.random.choice(len(P), N, replace=False)
        P = P[indices]
    elif len(P) < N:
        # Pad with random points if too few (unlikely with this s)
        # Add random points inside square
        while len(P) < N:
            P = np.vstack([P, np.random.uniform(-0.5, 0.5, 2)])
            
    P = P[:N]
    
    # Ensure valid start (no overlaps)
    # If overlaps exist, push them apart or just let SA handle it?
    # SA handles overlaps by accepting moves that increase r.
    # Initial r might be 0 if overlapping.
    
    current_P = P.copy()
    current_r = compute_radius(current_P)
    best_P = current_P.copy()
    best_r = current_r
    
    # 2. Simulated Annealing
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.9995
    
    while T > T_min:
        if time.time() > deadline - 10.0:
            break
            
        # Move
        i = np.random.randint(N)
        old_pos = current_P[i].copy()
        # Gaussian noise
        noise = np.random.normal(0, 0.01, 2)
        new_pos = current_P[i] + noise
        
        # Clip to boundary
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        current_P[i] = new_pos
        
        new_r = compute_radius(current_P)
        
        # Acceptance
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_P = current_P.copy()
        else:
            delta_r = current_r - new_r
            if delta_r > 0:
                prob = np.exp(-delta_r / T)
                if np.random.rand() < prob:
                    current_r = new_r
                else:
                    current_P[i] = old_pos
        
        T *= cooling_rate
        
    # 3. Polish with L-BFGS-B
    # Use the best_P found by SA
    P_opt = best_P.copy()
    
    # Define constraints to keep points in box
    # L-BFGS-B handles bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Flatten P for scipy
    x0 = P_opt.flatten()
    
    def obj_flat(x):
        P_flat = x.reshape(-1, 2)
        return soft_min_objective(P_flat, beta=0.01)
    
    try:
        res = minimize(obj_flat, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-10})
        if res.success:
            P_polished = res.x.reshape(-1, 2)
            r_polished = compute_radius(P_polished)
            if r_polished > best_r:
                best_P = P_polished
                best_r = r_polished
    except Exception:
        pass
        
    return best_P
