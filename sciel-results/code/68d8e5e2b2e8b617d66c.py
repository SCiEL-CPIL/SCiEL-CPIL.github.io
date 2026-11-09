# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize A_min / A_hull.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 620  # Leave 10s margin from 630s
    
    def triangle_areas(pts):
        """Compute all triangle areas from 13 points. Returns array of 286 areas."""
        # pts shape (13, 2)
        # For each triplet (i, j, k), area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_y)|
        idx = np.arange(n)
        i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
        # Only keep i < j < k
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_flat = i_idx[mask]
        j_flat = j_idx[mask]
        k_flat = k_idx[mask]
        
        xi = pts[i_flat, 0]
        yi = pts[i_flat, 1]
        xj = pts[j_flat, 0]
        yj = pts[j_flat, 1]
        xk = pts[k_flat, 0]
        yk = pts[k_flat, 1]
        
        areas = 0.5 * np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        return areas
    
    def hull_area(pts):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(pts)
            return hull.volume  # In 2D, volume is area
        except Exception:
            return 0.0
    
    def score(pts):
        """Compute A_min / A_hull."""
        areas = triangle_areas(pts)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        a_hull = hull_area(pts)
        if a_hull <= 0:
            return 0.0
        return a_min / a_hull
    
    def neg_score(pts_flat):
        """Negative score for minimization."""
        pts = pts_flat.reshape(n, 2)
        # Check for validity
        if not np.all(np.isfinite(pts)):
            return 1e10
        s = score(pts)
        return -s
    
    # Initial points: try several starting configurations
    best_pts = None
    best_score = -np.inf
    
    # Start with a regular 13-gon on a circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Also try a grid-like pattern
    rng = np.random.default_rng(seed=42)
    
    # Multiple restarts with different initializations
    initializations = [
        init_pts,  # Regular polygon
        rng.random((n, 2)),  # Random
        np.column_stack([np.cos(angles * 2), np.sin(angles * 2)]),  # Another pattern
    ]
    
    # Add some structured patterns
    # Square with center
    sq = np.array([
        [0, 0],
        [1, 0], [1, 1], [0, 1], [-1, 0], [-1, 1], [0, -1], [0, 1],
        [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [1, -1]
    ])
    initializations.append(sq)
    
    for init in initializations:
        if time.time() - start_time > time_limit - 30:
            break
        
        # Normalize initial points to be in a reasonable range
        init = init - init.mean(axis=0)
        init = init / (np.max(np.abs(init)) + 1e-10)
        
        # Use Nelder-Mead optimization
        try:
            result = minimize(
                neg_score,
                init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if time.time() - start_time > time_limit - 20:
                break
            
            opt_pts = result.x.reshape(n, 2)
            opt_score = -result.fun
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
        except Exception:
            continue
    
    # Try more random restarts if time permits
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() - start_time > time_limit - 30:
            break
        
        # Generate random initial points
        init = rng.random((n, 2))
        init = init - init.mean(axis=0)
        init = init / (np.max(np.abs(init)) + 1e-10)
        
        try:
            result = minimize(
                neg_score,
                init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if time.time() - start_time > time_limit - 15:
                break
            
            opt_pts = result.x.reshape(n, 2)
            opt_score = -result.fun
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerance if we have a good result
    if best_pts is not None and time.time() - start_time < time_limit - 10:
        try:
            result = minimize(
                neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if time.time() - start_time < time_limit - 5:
                opt_pts = result.x.reshape(n, 2)
                opt_score = -result.fun
                if opt_score > best_score:
                    best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # If we still don't have a valid result, fall back to regular polygon
    if best_pts is None:
        best_pts = init_pts.copy()
    
    # Ensure the result is valid
    if not np.all(np.isfinite(best_pts)):
        best_pts = init_pts.copy()
    
    return best_pts


# EVOLVE-BLOCK-END
