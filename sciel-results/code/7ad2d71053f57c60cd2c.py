# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    def compute_ratio(params):
        """Compute (d_min/d_max)^2 from flattened point coordinates."""
        pts = params.reshape((n, 2))
        dists = pdist(pts, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(params):
        """Objective to minimize: negative of (d_min/d_max)."""
        pts = params.reshape((n, 2))
        dists = pdist(pts, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return -(d_min / d_max)

    # Initial guess: 4x4 grid
    x = np.arange(4)
    y = np.arange(4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    
    best_points = grid_points.copy()
    best_ratio = compute_ratio(grid_points.flatten())
    
    # Try multiple optimization strategies from the grid
    optimization_strategies = [
        ('L-BFGS-B', {'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}),
        ('Powell', {'maxiter': 1000, 'ftol': 1e-14, 'xtol': 1e-14}),
        ('Nelder-Mead', {'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}),
    ]
    
    for method, options in optimization_strategies:
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(
                objective, 
                grid_points.flatten(), 
                method=method, 
                options=options
            )
            
            candidate_points = res.x.reshape((n, 2))
            candidate_ratio = compute_ratio(candidate_points.flatten())
            
            if candidate_ratio > best_ratio:
                best_points = candidate_points
                best_ratio = candidate_ratio
                
        except Exception:
            continue
    
    # Try hexagonal-like arrangements
    if time.time() < deadline - 60:
        # Create a hexagonal packing pattern for 16 points
        # Rows: 4, 4, 4, 4 with staggered positions
        hex_points = []
        row_spacing = np.sqrt(3) / 2
        for row in range(4):
            y_pos = row * row_spacing
            for col in range(4):
                x_pos = col + (0.5 if row % 2 == 1 else 0)
                hex_points.append([x_pos, y_pos])
        hex_points = np.array(hex_points)
        
        try:
            res_hex = minimize(
                objective, 
                hex_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            cand = res_hex.x.reshape((n, 2))
            ratio = compute_ratio(cand.flatten())
            if ratio > best_ratio:
                best_points = cand
                best_ratio = ratio
        except Exception:
            pass
    
    # Random restarts with optimization
    if time.time() < deadline - 45:
        np.random.seed(42)
        for i in range(8):
            if time.time() > deadline - 15:
                break
            
            # Generate random initial points
            rand_pts = np.random.randn(n, 2) * 2
            
            # Use Powell for better exploration
            try:
                res_rand = minimize(
                    objective, 
                    rand_pts.flatten(), 
                    method='Powell', 
                    options={'maxiter': 300, 'ftol': 1e-12, 'xtol': 1e-12}
                )
                cand = res_rand.x.reshape((n, 2))
                ratio = compute_ratio(cand.flatten())
                if ratio > best_ratio:
                    best_points = cand
                    best_ratio = ratio
            except Exception:
                continue
    
    # Final refinement with the best points found
    if time.time() < deadline - 10:
        try:
            res_final = minimize(
                objective, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-11}
            )
            cand = res_final.x.reshape((n, 2))
            ratio = compute_ratio(cand.flatten())
            if ratio > best_ratio:
                best_points = cand
                best_ratio = ratio
        except Exception:
            pass
    
    return best_points

# EVOLVE-BLOCK-END
