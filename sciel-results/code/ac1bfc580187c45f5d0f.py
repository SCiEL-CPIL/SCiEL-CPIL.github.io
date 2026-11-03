import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def compute_radius(centers):
        """Compute the radius efficiently."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        # Use KDTree for fast nearest neighbor
        tree = cKDTree(centers)
        # Query k=2 to get nearest neighbor (k=1 is self)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1])
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)
    
    def soft_min_objective(centers, beta=100.0):
        """
        Smooth approximation of the negative radius using log-sum-exp.
        We minimize the negative soft-min of distances.
        """
        wall_dists = 0.5 - np.abs(centers)
        
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        pair_dists = dists[:, 1]
        
        # Soft min of wall distances
        max_wall = np.max(wall_dists)
        log_sum_wall = np.log(np.sum(np.exp(wall_dists / beta - max_wall / beta))) + max_wall / beta
        soft_min_wall = beta * log_sum_wall
        
        # Soft min of pair distances
        max_pair = np.max(pair_dists)
        log_sum_pair = np.log(np.sum(np.exp(pair_dists / beta - max_pair / beta))) + max_pair / beta
        soft_min_pair = beta * log_sum_pair
        
        # We want to maximize min(soft_min_wall, soft_min_pair/2)
        # So we minimize the negative of a smooth min
        # Use smooth minimum: -beta * log(exp(-a/beta) + exp(-b/beta))
        a = soft_min_wall
        b = soft_min_pair / 2.0
        
        max_ab = np.max([a, b])
        log_sum = np.log(np.exp(-a / beta) + np.exp(-b / beta))
        smooth_min = -beta * log_sum + max_ab
        
        # Return negative to minimize
        return -smooth_min
    
    def hexagonal_initialization(n_rows, n_cols, jitter_scale=0.0):
        """Create hexagonal packing layout with optional jitter."""
        d = 1.0 / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        
        # Add jitter
        if jitter_scale > 0:
            jitter = np.random.randn(len(pts), 2) * jitter_scale
            pts = pts + jitter
            # Clip to bounds
            pts = np.clip(pts, -0.499, 0.499)
        
        return pts
    
    def nelder_mead_optimize(centers, max_iter=200, beta=50.0):
        """Optimize using Nelder-Mead on the soft-min objective."""
        # Flatten centers
        x0 = centers.flatten()
        
        # Define objective that reshapes
        def objective(x):
            c = x.reshape(-1, 2)
            # Check bounds
            if np.any(c < -0.5) or np.any(c > 0.5):
                return 1e10
            return soft_min_objective(c, beta)
        
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        return result.x.reshape(-1, 2)
    
    def final_polish(centers, max_rounds=20):
        """High-precision polish using direct distance maximization."""
        centers = centers.copy()
        current_r = compute_radius(centers)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 10:
                break
            
            improved = False
            tree = cKDTree(centers)
            
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                orig = centers[i].copy()
                best_pos = orig
                best_r = current_r
                
                # Find nearest neighbors
                dists, idxs = tree.query(orig, k=5)
                
                # Try small moves in 8 directions + random
                step = 0.001 / (round_num + 1)
                directions = [
                    (1, 0), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)
                ]
                # Normalize
                directions = [np.array(d) / np.linalg.norm(d) for d in directions]
                # Add some random directions
                for _ in range(4):
                    angle = np.random.rand() * 2 * np.pi
                    directions.append(np.array([math.cos(angle), math.sin(angle)]))
                
                for d in directions:
                    for step_mult in [0.5, 1.0, 2.0, 4.0]:
                        new_center = orig + d * step * step_mult
                        if new_center[0] < -0.5 or new_center[0] > 0.5:
                            continue
                        if new_center[1] < -0.5 or new_center[1] > 0.5:
                            continue
                        
                        centers[i] = new_center
                        new_r = compute_radius(centers)
                        
                        if new_r > best_r + 1e-12:
                            best_r = new_r
                            best_pos = new_center.copy()
                            improved = True
                
                centers[i] = best_pos
            
            current_r = best_r
            
            if not improved:
                break
        
        return centers, current_r
    
    # Generate diverse initial configurations
    best_centers = None
    best_radius = 0.0
    
    # Try multiple hexagonal layouts with different parameters
    configs = []
    for n_rows in range(13, 17):
        n_cols = math.ceil(N / n_rows)
        for jitter in [0.0, 0.001, 0.002, 0.003]:
            pts = hexagonal_initialization(n_rows, n_cols, jitter)
            if pts is not None:
                configs.append(pts)
    
    # Also try square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    configs.append(grid[:N])
    
    # Optimize each configuration
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        # Evaluate initial
        r = compute_radius(init_pts)
        if r > best_radius:
            best_radius = r
            best_centers = init_pts.copy()
        
        # Nelder-Mead optimization
        if time.time() < deadline - 25:
            try:
                optimized = nelder_mead_optimize(init_pts, max_iter=100, beta=50.0)
                r_opt = compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                pass
        
        # Final polish on best so far
        if time.time() < deadline - 15:
            polished, r_polished = final_polish(best_centers, max_rounds=10)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished.copy()
    
    # Final high-precision polish
    if best_centers is not None and time.time() < deadline - 5:
        polished, r_polished = final_polish(best_centers, max_rounds=15)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished.copy()
    
    if best_centers is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
