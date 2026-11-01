import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def compute_score(centers):
    # centers shape (N, 3)
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    max_coord = np.max(np.abs(centers))
    if max_coord > 0.5:
        return 0.0
    
    # Calculate min distance
    dists = cdist(centers, centers)
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Calculate min distance to wall
    min_wall_dist = 0.5 - np.max(np.abs(centers))
    
    r = min(min_dist / 2.0, min_wall_dist)
    
    if r < 0:
        return 0.0
        
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_centers = None
    best_score = 0.0
    
    # Strategy: Multi-start L-BFGS
    # We will try several different initial configurations and optimize each.
    
    def objective(params):
        c = params[:300].reshape(100, 3)
        r = params[300]
        
        # Penalty for boundary violation
        # We want |c_ik| <= 0.5 - r
        # Violation: max(0, |c_ik| - (0.5 - r))
        wall_viol = np.maximum(0, np.abs(c) - (0.5 - r))
        wall_pen = np.sum(wall_viol**2)
        
        # Penalty for overlap
        # We want |c_i - c_j| >= 2r
        # Violation: max(0, 2r - |c_i - c_j|)
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        overlap_viol = np.maximum(0, 2*r - dists)
        overlap_pen = np.sum(overlap_viol**2)
        
        # We want to maximize r, so minimize -r
        # Add penalties. The penalty weight needs to be high enough to enforce constraints
        # but not so high that it dominates the gradient of -r when constraints are satisfied.
        # A dynamic penalty or a fixed high weight works. Let's use a fixed weight.
        weight = 100.0
        return -r + weight * (wall_pen + overlap_pen)
        
    bounds = [(-0.5, 0.5)] * 300 + [(0.0, 0.5)]
    
    # Generate initial configurations
    inits = []
    
    # 1. Regular Grid (5x5x4)
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.3, 0.3, 4)
    grid_centers = np.array([(x[i], y[j], z[k]) for k in range(4) for j in range(5) for i in range(5)])
    inits.append(grid_centers)
    
    # 2. Random Uniform
    for i in range(5):
        rand_centers = np.random.uniform(-0.4, 0.4, (100, 3))
        inits.append(rand_centers)
        
    # 3. Fibonacci Sphere projected to cube (roughly)
    # This provides a more "even" distribution than pure random in some cases
    # We can use a simple spiral on a sphere and scale/clip, or just use random for simplicity 
    # since we have limited time. Let's stick to random and grid variations.
    
    # 4. Grid with noise
    for i in range(5):
        noise = np.random.normal(0, 0.05, (100, 3))
        noisy_grid = np.clip(grid_centers + noise, -0.45, 0.45)
        inits.append(noisy_grid)
        
    # Shuffle inits to mix strategies
    np.random.shuffle(inits)
    
    current_init_idx = 0
    current_opt_iter = 0
    
    while time.time() < deadline:
        # Select initial configuration
        if current_init_idx >= len(inits):
            # If we run out of predefined inits, generate new random ones
            rand_centers = np.random.uniform(-0.4, 0.4, (100, 3))
            inits.append(rand_centers)
            
        c_init = inits[current_init_idx]
        
        # Initial r estimate
        # Use a conservative estimate based on the initial configuration
        r_init = compute_score(c_init)
        if r_init <= 0:
             r_init = 0.05
             
        x0 = np.concatenate([c_init.flatten(), [r_init]])
        
        # Run optimization
        # We limit maxiter to prevent a single run from taking too long
        # L-BFGS-B is relatively fast per iteration but might need many
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        
        c_new = res.x[:300].reshape(100, 3)
        r_new = res.x[300]
        
        # Clip centers to valid range to ensure validity
        c_new = np.clip(c_new, -0.5, 0.5)
        
        score_new = compute_score(c_new)
        
        if score_new > best_score:
            best_score = score_new
            best_centers = c_new
            # If we found a good score, we might want to restart from this point with a different strategy
            # But for now, we just move to the next init.
            
        # Move to next initial configuration
        current_init_idx += 1
        
        # If we have done a few iterations from the same "type" of init, maybe cycle back?
        # For simplicity, we just iterate through the list. The list grows if we run out.
        
    if best_centers is None:
        # Fallback: return the first grid if nothing else worked
        return grid_centers
        
    return best_centers
