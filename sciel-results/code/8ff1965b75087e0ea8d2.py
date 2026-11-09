"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def construct():
    # 1. Generate a hexagonal lattice initialization
    # We aim for roughly 200 points. 
    # A hex grid with 14 rows alternating 15 and 14 points gives 203 points.
    # Rows: 0..13
    # Even rows (0,2,4,6,8,10,12): 15 points
    # Odd rows (1,3,5,7,9,11,13): 14 points
    # Total: 7*15 + 7*14 = 105 + 98 = 203
    
    rows = 14
    points = []
    # Spacing in hex grid
    s = 1.0
    h = s * np.sqrt(3) / 2
    
    for i in range(rows):
        if i % 2 == 0:
            n_points = 15
            offset = 0.0
        else:
            n_points = 14
            offset = s / 2.0
            
        for j in range(n_points):
            x = j * s + offset
            y = i * h
            points.append([x, y])
            
    points = np.array(points)
    
    # We have 203 points. We need to remove 3.
    # Remove points that are furthest from the center
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    sorted_indices = np.argsort(dists)
    points = points[sorted_indices[:N]]
    
    # 2. Scale and translate to fit [-0.5, 0.5]^2
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Scale to fit inside, leaving a small margin for safety
    scale = 1.0 / max(width, height) * 0.95
    
    # Center the points
    points = (points - center) * scale
    
    # 3. Simulated Annealing Optimization
    def get_radius(pts):
        # Calculate min distance to boundary
        dx = 0.5 - np.abs(pts[:, 0])
        dy = 0.5 - np.abs(pts[:, 1])
        min_boundary = np.min(np.minimum(dx, dy))
        
        # Calculate min pairwise distance
        # pdist is efficient for this size
        dists = pdist(pts)
        min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)

    current_pts = points.copy()
    current_r = get_radius(current_pts)
    
    # SA Parameters
    T = 0.005
    T_min = 1e-7
    alpha = 0.9998
    n_iter = 50000
    
    # Time limit check
    start_time = time.time()
    time_limit = 100.0 # Leave 20s buffer
    
    for it in range(n_iter):
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # Pick random point
        k = np.random.randint(N)
        
        # Propose move
        # Step size scales with temperature
        step_size = T * 5.0
        new_pos = current_pts[k].copy()
        new_pos += np.random.normal(0, step_size, 2)
        
        # Calculate new radius
        # We copy the array to avoid modifying current_pts until accepted
        # This is O(N) copy, which is fast
        temp_pts = current_pts.copy()
        temp_pts[k] = new_pos
        
        new_r = get_radius(temp_pts)
        
        # Acceptance criterion
        delta_r = new_r - current_r
        if delta_r > 0:
            current_pts = temp_pts
            current_r = new_r
        else:
            # Accept with probability exp(delta_r / T)
            # Handle underflow/overflow
            if T > 0:
                prob = np.exp(delta_r / T)
                if np.random.rand() < prob:
                    current_pts = temp_pts
                    current_r = new_r
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min

    return current_pts
# EVOLVE-BLOCK-END
