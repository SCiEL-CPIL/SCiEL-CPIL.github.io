# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

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

def simulated_annealing(points, time_limit):
    """
    Refine points using Simulated Annealing.
    """
    start_time = time.time()
    n = points.shape[0]
    current_points = points.copy()
    current_score = calculate_score(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    # SA Parameters
    T = 0.5  # Initial temperature
    alpha = 0.999 # Cooling rate
    min_T = 1e-5
    max_iters = 50000
    
    # We estimate the typical distance scale to set the step size
    # d_max is a good upper bound, d_min is lower.
    # Let's use a fraction of the bounding box diagonal or d_max.
    dists = spd.pdist(current_points)
    scale = np.max(dists) if len(dists) > 0 else 1.0
    step_size = 0.1 * scale
    
    for i in range(max_iters):
        if time.time() > start_time + time_limit:
            break
        if T < min_T:
            break
            
        # Generate a new candidate by perturbing one or a few points
        # Perturbing 1 point is often better for high dimensions to avoid noise
        idx = np.random.randint(n)
        candidate_points = current_points.copy()
        candidate_points[idx] += np.random.normal(0, step_size, 2)
        
        candidate_score = calculate_score(candidate_points)
        
        # Acceptance criterion
        if candidate_score > current_score:
            current_points = candidate_points
            current_score = candidate_score
            if current_score > best_score:
                best_points = candidate_points.copy()
                best_score = current_score
        else:
            # Accept with probability exp((new_score - current_score) / T)
            # Since we maximize, delta is negative.
            delta = candidate_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    current_points = candidate_points
                    current_score = candidate_score
            # If delta is 0, we don't really care, but usually distinct points have distinct scores
            # Note: if delta is positive, it's already accepted in the if block above.
            
        T *= alpha
        
        # Adaptive step size: if we are stuck, maybe increase step? 
        # Or decrease it as we cool? 
        # Usually step size is constant or decreases with T.
        # Let's keep it simple: constant step size scaled by initial scale.
        # But if best_score hasn't improved in a while, we might be stuck.
        
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
    
    # Generate a set of initializations
    inits = []
    
    # 1. Standard Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    points_grid = np.column_stack((xx.ravel(), yy.ravel()))
    inits.append(points_grid)
    
    # 2. Hexagonal-like Grid (shifted rows)
    # We want to pick 16 points from a hexagonal lattice that are well-distributed.
    # A 4x4 block in a hex lattice is a good start.
    pts_hex = []
    # Generate a larger hex lattice and pick a subset or just construct a 4x4 shifted one
    for i in range(4):
        for j in range(4):
            x_pos = j + (0.5 if i % 2 else 0)
            y_pos = i * np.sqrt(3) / 2
            pts_hex.append([x_pos, y_pos])
    pts_hex = np.array(pts_hex)
    pts_hex -= np.mean(pts_hex, axis=0)
    inits.append(pts_hex)
    
    # 3. Another Hexagonal variant (different shift or orientation)
    pts_hex2 = []
    for i in range(4):
        for j in range(4):
            x_pos = j * np.sqrt(3)/2 + (0.5 if i % 2 else 0) * np.sqrt(3)
            y_pos = i * 1.5 # Vertical spacing for hex
            pts_hex2.append([x_pos, y_pos])
    pts_hex2 = np.array(pts_hex2)
    pts_hex2 -= np.mean(pts_hex2, axis=0)
    inits.append(pts_hex2)
    
    # 4. Random
    np.random.seed(42)
    pts_rand = np.random.randn(n, d)
    pts_rand -= np.mean(pts_rand, axis=0)
    inits.append(pts_rand)
    
    # 5. Random
    np.random.seed(123)
    pts_rand2 = np.random.randn(n, d)
    pts_rand2 -= np.mean(pts_rand2, axis=0)
    inits.append(pts_rand2)
    
    best_points = inits[0]
    best_score = calculate_score(best_points)
    
    # Refine each initialization
    # We have ~590s. We have 5 inits.
    # Let's give each init ~100s if we run them sequentially? 
    # Or run them in parallel? No, Python is single-threaded here effectively for CPU bound.
    # Let's allocate time dynamically.
    
    num_inits = len(inits)
    # Reserve 20s for final polish
    time_per_init = (590 - 20) / num_inits
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # Calculate remaining time for this init, but cap it to time_per_init to ensure we try all
        # Actually, if one is very good, we might want to spend more time on it later.
        # For now, equal time.
        allocated_time = min(time_per_init, deadline - 20 - time.time())
        
        if allocated_time <= 0:
            break
            
        res_points = simulated_annealing(init_pts, allocated_time)
        res_score = calculate_score(res_points)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    # Final polish on the best result
    if time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        best_points = simulated_annealing(best_points, remaining_time)
        
    return best_points

# EVOLVE-BLOCK-END
