import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_SECONDS = 115  # Safety margin before 120s limit

def _score(points):
    """Compute the maximum valid radius for the given sphere centers."""
    if points.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points > 0.5) or np.any(points < -0.5):
        return 0.0
        
    # Distance between all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to walls: cube is [-0.5, 0.5]^3
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _objective_flat(x, beta=50.0):
    """
    Objective function for optimization: negative of a smooth approximation to the minimum radius.
    We minimize this, so we want to maximize the radius.
    Uses log-sum-exp for smoothness.
    """
    points = x.reshape(N, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(points)
    
    # Pair distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e10)
    
    # Convert to distances
    # We want min(d_ij/2, wall_dists)
    # Let's work with squared distances to avoid sqrt, but we need to be careful.
    # Actually, let's just compute the raw distances.
    dists = np.sqrt(dists_sq)
    
    # Collect all constraint values (distances that must be >= 2r)
    # Pair constraints: dists[i,j] >= 2r
    # Wall constraints: wall_dists[i,k] >= r
    
    # For a smooth approximation of min, we can use -1/beta * log(sum(exp(-beta * val)))
    # This approximates min(val) as beta -> inf
    
    # Pair values
    triu_idx = np.triu_indices(N, k=1)
    pair_dists = dists[triu_idx] / 2.0  # radius contribution from pair
    
    # Wall values
    wall_vals = wall_dists.flatten()  # radius contribution from wall
    
    # Combine all values
    all_vals = np.concatenate([pair_dists, wall_vals])
    
    # Smooth minimum
    # To avoid overflow, subtract max
    max_val = np.max(all_vals)
    smooth_min = max_val - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (all_vals - max_val))))
    
    return -smooth_min

def _init_grid(n):
    """Generate an initial grid-based configuration."""
    # Create a 4x4x4 grid (64 points) and select 50
    # Or a 3x3x3 grid (27) + some random?
    # Let's try a 4x4x4 grid with random jitter
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                points.append([
                    -0.5 + (i + 0.5) * (1.0 / 4.0),
                    -0.5 + (j + 0.5) * (1.0 / 4.0),
                    -0.5 + (k + 0.5) * (1.0 / 4.0)
                ])
    
    points = np.array(points)
    
    # Randomly select 50
    idx = np.random.choice(len(points), n, replace=False)
    points = points[idx]
    
    # Add small random jitter to break symmetry
    points += np.random.randn(n, 3) * 0.01
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _init_random(n):
    """Generate a random initial configuration."""
    points = np.random.uniform(-0.45, 0.45, (n, 3))
    return points

def _optimize(points, max_time):
    """
    Optimize the configuration using L-BFGS-B on the smooth objective.
    """
    start_time = time.time()
    x0 = points.flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_x = x0.copy()
    best_f = _objective_flat(x0)
    
    # Run L-BFGS-B
    try:
        res = minimize(
            _objective_flat,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_f:
            best_f = res.fun
            best_x = res.x
            
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > max_time:
        return best_x.reshape(N, 3), -best_f
    
    # Polish with a second run starting from the best point
    # Use a smaller beta for the final polish to get closer to the true min
    def _objective_flat_fine(x, beta=200.0):
        points = x.reshape(N, 3)
        wall_dists = 0.5 - np.abs(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, 1e10)
        dists = np.sqrt(dists_sq)
        triu_idx = np.triu_indices(N, k=1)
        pair_dists = dists[triu_idx] / 2.0
        wall_vals = wall_dists.flatten()
        all_vals = np.concatenate([pair_dists, wall_vals])
        max_val = np.max(all_vals)
        smooth_min = max_val - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (all_vals - max_val))))
        return -smooth_min
    
    try:
        res2 = minimize(
            _objective_flat_fine,
            best_x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res2.success or res2.fun < best_f:
            best_f = res2.fun
            best_x = res2.x
            
    except Exception:
        pass
    
    return best_x.reshape(N, 3), -best_f

def construct():
    """Construct the sphere centers."""
    start_time = time.time()
    best_points = None
    best_score = 0.0
    
    # Run multiple restarts with different initializations
    n_restarts = 0
    while time.time() - start_time < DEADLINE_SECONDS:
        n_restarts += 1
        
        # Alternate between grid and random init
        if n_restarts % 2 == 0:
            points = _init_grid(N)
        else:
            points = _init_random(N)
        
        # Optimize
        opt_points, opt_score = _optimize(points, 10.0)
        
        # Compute actual score
        actual_score = _score(opt_points)
        
        if actual_score > best_score:
            best_score = actual_score
            best_points = opt_points.copy()
            
        # If we have a good score, we can try to refine further
        if best_score > 0.13:
            # Try a few more refinements from the best point
            for _ in range(3):
                if time.time() - start_time > DEADLINE_SECONDS:
                    break
                refined_points, refined_score = _optimize(best_points, 5.0)
                actual_refined_score = _score(refined_points)
                if actual_refined_score > best_score:
                    best_score = actual_refined_score
                    best_points = refined_points.copy()
    
    if best_points is None:
        best_points = _init_grid(N)
        
    return best_points

if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_score(centers)}")
    print(f"Shape: {centers.shape}")
