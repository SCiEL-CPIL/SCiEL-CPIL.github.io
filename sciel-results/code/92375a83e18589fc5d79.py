"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 300

def construct():
    n = N
    time_budget = 110.0
    start_time = time.time()
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing for N points in unit circle
    # Area = pi * 1^2. Hex cell area = sqrt(3)/2 * s^2.
    # N * (sqrt(3)/2 * s^2) ~ pi  => s ~ sqrt(2*pi / (N*sqrt(3)))
    s = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
    
    # Generate hex grid points
    pts_list = []
    # Estimate number of rows/cols to cover unit circle
    max_r = 1.0
    # Hex grid coordinates
    # We'll generate a large enough grid and filter
    range_r = int(np.ceil(max_r / s)) + 2
    for i in range(-range_r, range_r + 1):
        for j in range(-range_r, range_r + 1):
            # Hex to cartesian
            # Using pointy-top hexes or flat-top? Let's use standard axial to pixel
            # x = s * (j + i/2)
            # y = s * (sqrt(3)/2 * i)
            x = s * (j + i / 2.0)
            y = s * (np.sqrt(3) / 2.0 * i)
            if x*x + y*y <= max_r*max_r:
                pts_list.append([x, y])
                
    points = np.array(pts_list)
    
    # If we have more than n points, randomly sample n
    if len(points) > n:
        indices = np.random.choice(len(points), n, replace=False)
        points = points[indices]
    # If we have less, add random points (should not happen with range_r)
    elif len(points) < n:
        while len(points) < n:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) <= 1.0:
                points = np.vstack([points, p])

    # 2. Simulated Annealing
    # Score function: min( min_pair_dist/2, min(1 - |p|) )
    
    def calc_score(pts):
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Compute boundary distances
        norms = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)

    best_points = points.copy()
    best_score = calc_score(best_points)
    current_points = points.copy()
    current_score = best_score
    
    T = 0.01 # Initial temperature
    T_min = 1e-5
    alpha = 0.995 # Cooling rate
    
    # Optimization loop
    while time.time() - start_time < time_budget:
        if T < T_min:
            # Reheat or stop? Let's try to reheat slightly if stuck, or just break
            if np.random.rand() < 0.05:
                T = 0.001
            else:
                break
                
        # Generate a move: perturb 1-5 random points
        num_perturb = np.random.randint(1, 6)
        indices = np.random.choice(n, num_perturb, replace=False)
        
        # Step size scales with temperature
        step_size = T * 0.5
        new_points = current_points.copy()
        for idx in indices:
            new_points[idx] += np.random.normal(0, step_size, 2)
            
        # Ensure new points stay inside container (hard constraint)
        # If a point goes outside, project it back or reject? 
        # Projecting is better to maintain feasibility
        norms = np.linalg.norm(new_points, axis=1)
        outside_mask = norms > 1.0
        if np.any(outside_mask):
            new_points[outside_mask] /= norms[outside_mask, np.newaxis]
            
        new_score = calc_score(new_points)
        
        # Metropolis Criterion
        delta = new_score - current_score
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= alpha
        
    return best_points
# EVOLVE-BLOCK-END
