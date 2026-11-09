# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Leave 50s margin from 630s to be safe
    
    # Pre-compute indices for all 286 triangles
    # This is done once to save time in the inner loop
    idx = np.array(list(map(lambda x: x, range(286)))) # Placeholder, will compute properly
    # Efficient way to get indices
    from itertools import combinations
    tri_indices = np.array(list(combinations(range(n), 3)))
    i_idx = tri_indices[:, 0]
    j_idx = tri_indices[:, 1]
    k_idx = tri_indices[:, 2]
    
    def calculate_score(points):
        """
        Calculate A_min / A_hull.
        Returns 0 if invalid.
        """
        if not np.all(np.isfinite(points)):
            return 0.0
        
        # Calculate convex hull area
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 0:
                return 0.0
        except Exception:
            return 0.0

        # Calculate min triangle area
        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        # Cross product magnitude in 2D
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 0:
            return 0.0
            
        return A_min / A_hull

    def neg_objective(points_flat):
        """
        Negated objective for minimization.
        """
        points = points_flat.reshape((n, 2))
        score = calculate_score(points)
        return -score

    # Strategy: Simulated Annealing
    # 1. Generate diverse initial candidates (Random, Grid-based, Hexagonal-like)
    # 2. Run SA on each candidate
    # 3. Polish the best result with Nelder-Mead
    
    best_points_global = None
    best_score_global = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Define initial configurations
    initial_configs = []
    
    # 1. Random points
    initial_configs.append(rng.random((n, 2)))
    
    # 2. Grid-like points (4x4 grid minus one corner, slightly perturbed)
    grid_x, grid_y = np.meshgrid(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 4))
    grid_pts = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    # Remove one point to get 13? 4x4=16. Remove 3 corners?
    # Let's just take the first 13
    initial_configs.append(grid_pts[:13] + rng.random((13, 2)) * 0.05)
    
    # 3. Hexagonal packing approximation
    # Center point + rings
    hex_pts = []
    hex_pts.append([0.5, 0.5])
    # Ring 1 (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        hex_pts.append([0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)])
    # Ring 2 (12 points, we need 13 total, so 6 more)
    for i in range(6):
        angle = i * np.pi / 3 + np.pi/6
        hex_pts.append([0.5 + 0.6 * np.cos(angle), 0.5 + 0.6 * np.sin(angle)])
    # We have 1 + 6 + 6 = 13 points
    initial_configs.append(np.array(hex_pts))
    
    # Add a few more random perturbations of the hexagonal config
    for _ in range(3):
        perturbed = initial_configs[2] + rng.normal(0, 0.1, (n, 2))
        # Ensure within [0,1]
        perturbed = np.clip(perturbed, 0.0, 1.0)
        initial_configs.append(perturbed)

    # Simulated Annealing Parameters
    T_start = 0.1
    T_end = 1e-5
    cooling_rate = 0.99
    max_iterations = 2000
    
    for init_idx, init_points in enumerate(initial_configs):
        if time.time() - start_time > time_limit - 60: # Break if running out of time for remaining configs
            break
            
        current_points = init_points.copy()
        current_score = calculate_score(current_points)
        
        if current_score > best_score_global:
            best_score_global = current_score
            best_points_global = current_points.copy()
            
        if current_score <= 0:
            continue
            
        T = T_start
        for it in range(max_iterations):
            if time.time() - start_time > time_limit - 30:
                break
                
            # Generate neighbor: perturb a random point
            neighbor_points = current_points.copy()
            point_idx = rng.integers(0, n)
            # Step size depends on temperature
            step_size = T * 0.5
            neighbor_points[point_idx] += rng.normal(0, step_size, 2)
            # Keep within bounds
            neighbor_points[point_idx] = np.clip(neighbor_points[point_idx], 0.0, 1.0)
            
            neighbor_score = calculate_score(neighbor_points)
            
            if neighbor_score <= 0:
                # Invalid neighbor, reject
                # Optionally try smaller step or skip
                pass
            else:
                delta = neighbor_score - current_score
                if delta > 0:
                    # Accept better solution
                    current_points = neighbor_points
                    current_score = neighbor_score
                    if current_score > best_score_global:
                        best_score_global = current_score
                        best_points_global = current_points.copy()
                else:
                    # Accept worse solution with probability exp(delta / T)
                    # delta is negative, so exp(negative) < 1
                    prob = np.exp(delta / T)
                    if rng.random() < prob:
                        current_points = neighbor_points
                        current_score = neighbor_score
            
            T *= cooling_rate
            if T < T_end:
                break
        
        # Polish the best found so far in this SA run with Nelder-Mead
        if best_points_global is not None:
            # Use the best points found globally as the start for NM
            # Or polish the current best from this SA run? 
            # Let's polish the global best to ensure we are improving the overall champion
            try:
                res = minimize(
                    neg_objective, 
                    best_points_global.flatten(), 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                polished_points = res.x.reshape((n, 2))
                polished_score = calculate_score(polished_points)
                
                if polished_score > best_score_global:
                    best_score_global = polished_score
                    best_points_global = polished_points
            except Exception:
                pass

    # Final check and return
    if best_points_global is None or not np.all(np.isfinite(best_points_global)):
        # Fallback
        return rng.random((n, 2))
        
    return best_points_global

# EVOLVE-BLOCK-END
