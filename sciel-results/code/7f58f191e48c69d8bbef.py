# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd

def calculate_score(points):
    """
    Calculate the metric R = (d_min / d_max)^2
    """
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def local_search(points, time_limit):
    """
    Perform local search with simulated annealing to maximize the score.
    """
    n = points.shape[0]
    current_points = points.copy()
    current_score = calculate_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    
    # Simulated Annealing parameters
    temp = 0.01
    min_temp = 1e-6
    cooling_rate = 0.999
    
    # Step size for perturbations
    step_size = 0.1
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = np.random.randint(n)
        
        # Generate a random perturbation
        # We scale the perturbation based on the current spread of points
        # to ensure we explore the relevant scale
        spread = np.std(current_points)
        if spread == 0:
            spread = 1.0
            
        # Perturbation magnitude
        delta = np.random.randn(2) * step_size * spread
        
        new_points = current_points.copy()
        new_points[idx] += delta
        
        new_score = calculate_score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Simulated annealing acceptance
            if temp > min_temp:
                delta_score = new_score - current_score
                # Probability of accepting a worse solution
                # Since scores are small (e.g., 0.05), temp should be small
                prob = np.exp(delta_score / temp)
                if np.random.rand() < prob:
                    current_points = new_points
                    current_score = new_score
        
        # Cool down
        temp *= cooling_rate
        
        # Adaptive step size: if we've been improving, keep step size.
        # If stuck, maybe reduce step size to fine-tune.
        # For simplicity, we keep it constant or slowly reduce it.
        # Let's reduce step size slowly to converge.
        step_size *= 0.9999
        
    return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    
    # Deadline guard: 630s limit, use 600s for safety
    deadline = time.time() + 590
    
    # Initial guess: 4x4 grid
    # This provides a good baseline ratio
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Run local search
    # We can run multiple restarts if time permits
    best_points = points
    best_score = calculate_score(points)
    
    # Try a few different initializations
    inits = []
    
    # 1. Standard Grid
    inits.append(points)
    
    # 2. Hexagonal-like Grid (shifted rows)
    # Rows of 4 points, alternating shifts
    pts_hex = []
    for i in range(4):
        for j in range(4):
            x_pos = j + (0.5 if i % 2 else 0)
            y_pos = i * np.sqrt(3) / 2
            pts_hex.append([x_pos, y_pos])
    pts_hex = np.array(pts_hex)
    # Center and normalize
    pts_hex -= np.mean(pts_hex, axis=0)
    inits.append(pts_hex)
    
    # 3. Random
    np.random.seed(42)
    pts_rand = np.random.randn(n, d)
    pts_rand -= np.mean(pts_rand, axis=0)
    inits.append(pts_rand)
    
    # Allocate time for each init
    num_inits = len(inits)
    time_per_init = (time.time() + 590 - time.time()) / num_inits
    
    for init_pts in inits:
        if time.time() > deadline - 10:
            break
            
        # Run local search on this init
        # Give it a portion of the time
        end_time_for_this = time.time() + time_per_init
        res_points = local_search(init_pts, end_time_for_this - time.time())
        res_score = calculate_score(res_points)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    # If time remains, do a final polish on the best result
    if time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        best_points = local_search(best_points, remaining_time)
        
    return best_points

# EVOLVE-BLOCK-END
