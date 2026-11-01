import numpy as np
import time

N = 50

def compute_score(points):
    """
    Computes the radius r for a given set of points.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    """
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check bounds
    # Points must be in [-0.5, 0.5]
    if np.any(points < -0.5) or np.any(points > 0.5):
        return 0.0

    # Compute pairwise distances
    # Using broadcasting for speed
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Compute distance to walls
    # x distance to wall: 0.5 - abs(x)
    # y distance to wall: 0.5 - abs(y)
    wall_dists = np.min(np.stack([0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1])], axis=1), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def construct():
    start_time = time.time()
    deadline = 115.0  # 5 seconds safety margin
    
    # Initial configuration: Jittered grid
    # 7x7 grid covers 49 points. We need 50.
    # Let's use a 7x7 grid for the first 49 and place the 50th randomly, or just use 8x8 and take 50?
    # A 7x7 grid has r ~ 0.0714.
    # Let's start with a 7x7 grid + 1 random point, or a slightly distorted 7x7 grid.
    # Actually, let's just use a random start or a simple grid start.
    # A 7x7 grid start is good because it's close to optimal.
    
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # We need 50 points.
    # If we use 7x7, we have 49.
    # Let's add a 50th point at the center (0,0) and push others? 
    # Or just start with a random configuration to let SA do the work.
    # Random start is safer against getting stuck in a bad local minimum of the grid.
    # But grid start is faster to converge.
    # Let's try a jittered 7x7 grid + 1 extra point placed randomly.
    
    points = grid.copy()
    # Add 50th point
    # Place it randomly
    p50 = np.random.uniform(-0.5, 0.5, 2)
    points = np.vstack([points, p50])
    
    # Shuffle to ensure random order for SA moves
    np.random.shuffle(points)
    
    # Jitter the grid points slightly to break symmetry
    points += np.random.normal(0, 0.01, points.shape)
    
    # Clip to valid range
    points = np.clip(points, -0.5, 0.5)
    
    best_points = points.copy()
    best_score = compute_score(best_points)
    
    # Simulated Annealing Parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    alpha = 0.999  # Cooling rate
    step_size = 0.05  # Initial step size
    
    current_points = best_points.copy()
    current_score = best_score
    
    iters = 0
    while time.time() - start_time < deadline:
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Generate a move
        # Gaussian move
        move = np.random.normal(0, step_size, 2)
        new_points = current_points.copy()
        new_points[idx] += move
        
        # Keep in bounds
        new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
        
        new_score = compute_score(new_points)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        # Cooling
        T *= alpha
        # Occasionally reheat or adjust step size?
        # Keep it simple.
        
        iters += 1
        
        # Check time every 1000 iters to avoid overhead
        if iters % 1000 == 0:
            if time.time() - start_time > deadline - 5:
                break

    # Optional: Local refinement
    # Try to improve the best points found using a local search
    # We can try to move points that are involved in the minimum distance
    
    # For now, return the best found
    return best_points
