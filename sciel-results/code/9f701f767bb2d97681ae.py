import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 550  # Leave 80s margin from 630s to be safe

    # Pre-compute indices for all 286 triangles
    # 13 points -> C(13, 3) = 13*12*11 / 6 = 286
    indices = np.array(list(zip(*np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')))).reshape(-1, 3)
    # Filter unique combinations i < j < k
    mask = (indices[:, 0] < indices[:, 1]) & (indices[:, 1] < indices[:, 2])
    indices = indices[mask]
    
    def objective(x):
        """
        Minimize negative ratio A_min / A_hull.
        x: flat array of 26 coordinates.
        """
        P = x.reshape(n, 2)
        
        # Calculate triangle areas
        pi = P[indices[:, 0]]
        pj = P[indices[:, 1]]
        pk = P[indices[:, 2]]
        
        # Cross product z-component: (pj - pi) x (pk - pi)
        dx1 = pj[:, 0] - pi[:, 0]
        dy1 = pj[:, 1] - pi[:, 1]
        dx2 = pk[:, 0] - pi[:, 0]
        dy2 = pk[:, 1] - pi[:, 1]
        
        cross = dx1 * dy2 - dy1 * dx2
        areas = 0.5 * np.abs(cross)
        
        a_min = np.min(areas)
        
        if a_min < 1e-9:
            return 1e6  # Large penalty for collinear points
        
        # Calculate hull area
        try:
            hull = ConvexHull(P)
            a_hull = hull.volume  # In 2D, volume is area
            if a_hull < 1e-9:
                return 1e6
        except Exception:
            return 1e6
            
        ratio = a_min / a_hull
        return -ratio

    # Bounds for DE
    bounds = [(0, 1) for _ in range(2 * n)]
    
    # Differential Evolution
    # popsize=20, maxiter=50 is a reasonable start. 
    # We need to monitor time.
    best_x = None
    best_val = 1e6
    
    # Run DE
    try:
        # tol=1e-6, maxiter=100, popsize=20
        # Using callback to check time? DE doesn't support callback for stopping mid-run easily in older scipy, 
        # but we can just set maxiter conservatively.
        # 26 variables. 
        # Let's try maxiter=50, popsize=30.
        res = differential_evolution(objective, bounds, seed=42, maxiter=50, popsize=30, tol=1e-5, polish=False)
        best_x = res.x
        best_val = res.fun
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
             return best_x.reshape(n, 2)
             
    except Exception:
        # Fallback to random if DE fails
        best_x = np.random.rand(2 * n)
        
    # Local Polish with Nelder-Mead
    # This helps refine the solution found by DE.
    try:
        res_local = minimize(objective, best_x, method='Nelder-Mead', 
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res_local.fun < best_val:
            best_x = res_local.x
            best_val = res_local.fun
    except Exception:
        pass
        
    final_points = best_x.reshape(n, 2)
    
    # Final validity check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-9:
             # Fallback to random if invalid
             final_points = np.random.rand(n, 2)
    except Exception:
        final_points = np.random.rand(n, 2)

    return final_points
