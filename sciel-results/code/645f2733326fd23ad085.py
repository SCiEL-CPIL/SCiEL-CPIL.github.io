# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Maximizes the ratio of minimum to maximum pairwise distance for 16 points in 2D.
    Uses Differential Evolution for global search and Nelder-Mead for local refinement.
    """
    n = 16
    d = 2
    
    def objective(params):
        pts = params.reshape(n, d)
        
        # Center the points
        pts = pts - np.mean(pts, axis=0)
        
        # Check for degenerate case (all points same)
        ss = np.sum(pts**2)
        if ss < 1e-12:
            return 1.0 # Bad score (minimize, so large value is bad)
        
        # Normalize scale to fix scale invariance
        pts = pts / np.sqrt(ss)
        
        # Calculate pairwise distances
        dists = cdist(pts, pts)
        
        # Get upper triangle indices
        iu = np.triu_indices(n, k=1)
        pdists = dists[iu]
        
        d_min = np.min(pdists)
        d_max = np.max(pdists)
        
        if d_max < 1e-12:
            return 1.0
            
        ratio = d_min / d_max
        # We want to maximize ratio, so minimize -ratio
        return -ratio

    # Bounds for Differential Evolution
    # Since we normalize inside, the absolute scale doesn't matter, 
    # but the relative positions do. 
    # A reasonable range for normalized coordinates is roughly [-2, 2]
    bounds = [(-2, 2)] * (n * d)
    
    # Run Differential Evolution
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=500, 
            popsize=15, 
            tol=1e-8,
            polish=True # Uses L-BFGS-B to polish the best result
        )
        best_points = res.x.reshape(n, d)
    except Exception:
        # Fallback to grid if DE fails
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack((xx.ravel(), yy.ravel()))
        
    # Refine with Nelder-Mead for higher precision if needed
    try:
        res_local = minimize(
            objective, 
            best_points.ravel(), 
            method='Nelder-Mead', 
            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000}
        )
        if res_local.fun < objective(best_points.ravel()):
            best_points = res_local.x.reshape(n, d)
    except Exception:
        pass

    # Final cleanup: center and normalize
    best_points -= np.mean(best_points, axis=0)
    ss = np.sum(best_points**2)
    if ss > 1e-12:
        best_points /= np.sqrt(ss)
        
    if not np.all(np.isfinite(best_points)):
        # Fallback
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack((xx.ravel(), yy.ravel()))
        
    return best_points

# EVOLVE-BLOCK-END
