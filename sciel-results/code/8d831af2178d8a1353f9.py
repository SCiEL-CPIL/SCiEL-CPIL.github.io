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

def objective_function(flat_points):
    """
    Objective function for minimization: minimize d_max / d_min
    which is equivalent to maximizing d_min / d_max
    """
    points = flat_points.reshape(-1, 2)
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 1e10
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_min == 0:
        return 1e10
    return d_max / d_min

def refine_points(points, time_limit):
    """
    Refine points using Nelder-Mead optimization.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = calculate_score(best_points)
    
    # Flatten points for optimizer
    flat_points = points.flatten()
    
    # Nelder-Mead parameters
    # maxiter needs to be large enough to converge, but we check time
    # We'll run minimize with a reasonable maxiter and rely on time checks if it's too slow,
    # but Nelder-Mead is usually fast for 32 dimensions.
    
    try:
        # Use Nelder-Mead
        # xatol and fatol help with convergence speed
        result = minimize(
            objective_function, 
            flat_points, 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        
        if result.success or result.fun < 1e10:
            candidate_points = result.x.reshape(-1, 2)
            candidate_score = calculate_score(candidate_points)
            
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
    except Exception:
        pass
        
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
    # Nelder-Mead is fast, so we can try all of them.
    for init_pts in inits:
        if time.time() > deadline - 10:
            break
            
        # Refine
        res_points = refine_points(init_pts, 50) # 50s limit per refine, but NM is usually faster
        res_score = calculate_score(res_points)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    # Final polish on the best result
    if time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        best_points = refine_points(best_points, remaining_time)
        
    return best_points

# EVOLVE-BLOCK-END
