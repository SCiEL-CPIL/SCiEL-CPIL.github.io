import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    start_time = time.time()
    max_time = 600.0
    safety_margin = 15.0  # Increased margin for safety

    def get_score(points):
        # Calculate pairwise distances
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def local_search(points, max_iters=20000):
        """
        Hybrid local search:
        1. Coordinate-wise fine-tuning (x, y steps)
        2. Random perturbations to escape local minima
        Uses adaptive step sizes.
        """
        best_points = points.copy()
        best_score = get_score(best_points)
        
        # Initial step sizes
        step_coord = 0.05
        step_rand = 0.1
        
        stagnant = 0
        no_improve_count = 0
        
        for i in range(max_iters):
            if time.time() - start_time > max_time - safety_margin:
                break
                
            improved = False
            
            # Phase 1: Coordinate-wise moves (fast, precise)
            for j in range(n):
                # Try x-direction moves
                for dx in [-step_coord, step_coord]:
                    candidate = best_points.copy()
                    candidate[j, 0] += dx
                    new_score = get_score(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = candidate
                        improved = True
                        no_improve_count = 0
                        break # Move to next point if improved? Or continue? 
                        # Usually greedy: take first improvement. Let's break inner loop for this axis.
                
                # Try y-direction moves
                for dy in [-step_coord, step_coord]:
                    candidate = best_points.copy()
                    candidate[j, 1] += dy
                    new_score = get_score(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = candidate
                        improved = True
                        no_improve_count = 0
                        break

            # Phase 2: Random perturbation (if stagnating or periodically)
            # Every 10 iterations, or if no coordinate improvement, try a random kick
            if i % 10 == 0 or not improved:
                for _ in range(3):
                    perturbation = np.random.normal(0, step_rand, size=d)
                    candidate = best_points.copy()
                    candidate[np.random.randint(n)] += perturbation
                    new_score = get_score(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = candidate
                        improved = True
                        no_improve_count = 0

            if not improved:
                stagnant += 1
                no_improve_count += 1
                if stagnant > 5:
                    # Reduce step sizes
                    step_coord *= 0.8
                    step_rand *= 0.8
                    stagnant = 0
                    if step_coord < 1e-7 and step_rand < 1e-7:
                        break
            else:
                stagnant = 0
                # Slightly increase step size if improving, capped
                step_coord = min(step_coord * 1.02, 0.2)
                step_rand = min(step_rand * 1.02, 0.5)
                
        return best_points, best_score

    # Initial configurations
    configs = []
    
    # 1. 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_points = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    configs.append(grid_points)
    
    # 2. Regular 16-gon
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    poly_points = np.column_stack((np.cos(angles), np.sin(angles)))
    configs.append(poly_points)
    
    # 3. Hexagonal-like packing
    hex_pts = []
    for row in range(4):
        for col in range(4):
            x_h = col * 1.0 + (0.5 if row % 2 == 1 else 0)
            y_h = row * 0.866
            hex_pts.append([x_h, y_h])
    hex_pts = np.array(hex_pts)
    hex_pts -= hex_pts.mean(axis=0)
    configs.append(hex_pts)
    
    # 4. Two concentric squares (8+8)
    outer = np.array([
        [-1, -1], [1, -1], [1, 1], [-1, 1],
        [0, -1], [1, 0], [0, 1], [-1, 0]
    ])
    inner_scale = 0.5
    inner = outer * inner_scale
    theta = np.pi / 4
    c, s = np.cos(theta), np.sin(theta)
    R_rot = np.array([[c, -s], [s, c]])
    inner = inner @ R_rot.T
    two_squares = np.vstack([outer, inner])
    configs.append(two_squares)
    
    # 5. Random configuration
    np.random.seed(42)
    configs.append(np.random.randn(n, d) * 2)
    
    # 6. Rotated Grid
    grid_rot = grid_points.copy()
    grid_rot = grid_rot @ R_rot.T
    configs.append(grid_rot)
    
    # 7. Grid with corners pulled in
    grid2 = grid_points.copy()
    corners = [0, 3, 12, 15]
    for idx in corners:
        grid2[idx] *= 0.8
    configs.append(grid2)
    
    best_points_global = configs[0]
    best_score_global = get_score(configs[0])
    
    # Run local search from each configuration
    for config_idx, config in enumerate(configs):
        if time.time() - start_time > max_time - safety_margin:
            break
        
        # Run multiple times with different random seeds for perturbation
        # We use 2 runs per config to balance exploration and time
        for run in range(2):
            if time.time() - start_time > max_time - safety_margin:
                break
            
            # Seed based on config and run for reproducibility
            np.random.seed(1000 + config_idx * 10 + run)
            pts, score = local_search(config.copy())
            
            if score > best_score_global:
                best_score_global = score
                best_points_global = pts.copy()
                
    return best_points_global

# EVOLVE-BLOCK-END
