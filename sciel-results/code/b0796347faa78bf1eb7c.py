import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:N].astype(np.float64)
    
    def evaluate(pts):
        # Calculate min distance between pairs
        # and min distance to walls
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Wall distances
        dist_x = 0.5 - np.abs(pts[:, 0])
        dist_y = 0.5 - np.abs(pts[:, 1])
        min_wall_dist = np.min(np.minimum(dist_x, dist_y))
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return r

    current_r = evaluate(points)
    best_points = points.copy()
    best_r = current_r
    
    # Simulated Annealing parameters
    T_initial = 0.01  
    T_min = 1e-6
    cooling = 0.999
    base_step_size = 0.02
    
    T = T_initial
    
    while time.time() < deadline:
        # Pick a random point
        i = np.random.randint(N)
        
        # Generate a perturbation
        current_step = base_step_size * (T / T_initial)
        
        dx = np.random.normal(0, current_step)
        dy = np.random.normal(0, current_step)
        
        new_points = points.copy()
        new_points[i, 0] += dx
        new_points[i, 1] += dy
        
        # Quick check: if it goes far outside, reject early?
        nx, ny = new_points[i]
        if nx < -0.5 or nx > 0.5 or ny < -0.5 or ny > 0.5:
             # It's outside. r will be negative.
             T *= cooling
             continue
             
        new_r = evaluate(new_points)
        
        # Acceptance
        if new_r > current_r or np.random.rand() < np.exp((new_r - current_r) / T):
            points = new_points
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_points
