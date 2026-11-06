import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 50

def calculate_min_distance(x):
    """
    Calculate the minimum distance between any pair of circles 
    and the minimum distance from any circle to the square boundary.
    x: (N, 2) array of centers.
    Returns: min distance (which is 2*r).
    """
    # Pairwise distances
    if N > 1:
        d_pairs = pdist(x, 'euclidean')
        min_pair = np.min(d_pairs)
    else:
        min_pair = 1.0
        
    # Distance to walls
    # Square is [-0.5, 0.5] x [-0.5, 0.5]
    # dist to left: x[:, 0] - (-0.5) = x[:, 0] + 0.5
    # dist to right: 0.5 - x[:, 0]
    # dist to bottom: x[:, 1] + 0.5
    # dist to top: 0.5 - x[:, 1]
    
    d_left = x[:, 0] + 0.5
    d_right = 0.5 - x[:, 0]
    d_bottom = x[:, 1] + 0.5
    d_top = 0.5 - x[:, 1]
    
    # Ensure points are inside? 
    # If a point is outside, the distance to the "wall" it crossed is negative?
    # The score definition: min_i min_k (1/2 - |c_ik|).
    # If c_ik > 0.5, 1/2 - c_ik is negative.
    # So we just take the min of these 4 values for each point.
    
    d_walls = np.min(np.array([d_left, d_right, d_bottom, d_top]), axis=0)
    min_wall = np.min(d_walls)
    
    return min(min_pair, min_wall)

def generate_hex_initialization(n_points):
    """
    Generate a hexagonal lattice of points and select the n_points closest to the center.
    """
    # Generate a grid of hex points
    # Spacing s
    s = 0.15
    h = s * np.sqrt(3) / 2
    
    # Create a range of x and y
    x_range = np.arange(-1.0, 1.0, s/2) # Fine grid for generation
    # Actually, just generate rows
    points = []
    y = -1.0
    row_idx = 0
    while y <= 1.0:
        x_offset = (s/2) if (row_idx % 2 == 1) else 0
        x = -1.0 + x_offset
        while x <= 1.0:
            points.append((x, y))
            x += s
        y += h
        row_idx += 1
            
    points = np.array(points)
    
    # Filter points that are roughly inside or near the square
    # We want to pick points that can fit inside.
    # Let's pick the n_points closest to the origin (0,0)
    # This creates a compact cluster.
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    selected = points[indices[:n_points]]
    
    # Center the selected points?
    # They are already centered around 0 roughly.
    # But we should ensure they are within [-0.5, 0.5] initially?
    # The optimizer will push them.
    # However, if they start outside, NM might struggle with the non-differentiability at the boundary?
    # The objective handles it.
    
    # Add a small random jitter to break symmetry
    selected += np.random.normal(0, 0.001, selected.shape)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Initial Guess: Hexagonal Lattice
    x0 = generate_hex_initialization(N)
    best_x = x0
    best_score = calculate_min_distance(x0)
    
    def objective(x_flat):
        x = x_flat.reshape(-1, 2)
        return -calculate_min_distance(x)
        
    # 2. Optimization with Nelder-Mead
    # We will run this a few times with different initializations
    
    initializations = [x0]
    
    # Add a square grid initialization (perturbed)
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N] + np.random.normal(0, 0.01, (N, 2))
    initializations.append(grid)
    
    # Add a random initialization
    x_rand = np.random.uniform(-0.45, 0.45, (N, 2))
    initializations.append(x_rand)
    
    for i, x_init in enumerate(initializations):
        if time.time() > deadline:
            break
            
        # Ensure x_init is a copy
        x_current = x_init.copy()
        
        try:
            # Run Nelder-Mead
            # maxiter 200-500 is usually enough for local refinement
            # We use a larger maxiter for the first (hex) run
            max_iter = 500 if i == 0 else 200
            
            res = minimize(objective, x_current.flatten(), method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
            
            if res.x is not None:
                x_opt = res.x.reshape(-1, 2)
                score_opt = calculate_min_distance(x_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_x = x_opt
        except Exception as e:
            pass
            
        # If time allows, do a second pass on the best result found so far
        if i == 0 and time.time() < deadline - 20:
             # Perturb best_x
             x_pert = best_x + np.random.normal(0, 0.02, best_x.shape)
             try:
                 res = minimize(objective, x_pert.flatten(), method='Nelder-Mead', 
                                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
                 if res.x is not None:
                     x_opt = res.x.reshape(-1, 2)
                     score_opt = calculate_min_distance(x_opt)
                     if score_opt > best_score:
                         best_score = score_opt
                         best_x = x_opt
             except Exception as e:
                 pass

    # Final check: ensure valid shape and finite values
    if best_x.shape != (N, 2):
        return np.zeros((N, 2))
        
    if not np.all(np.isfinite(best_x)):
        return np.zeros((N, 2))
        
    return best_x
