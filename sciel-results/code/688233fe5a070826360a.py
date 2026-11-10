import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
        n = 16
        
        # Define a function to evaluate the score
        def score(points):
            d = pdist(points)
            d_min = np.min(d)
            d_max = np.max(d)
            if d_max == 0:
                return 0.0
            return (d_min / d_max) ** 2
            
        # Define the objective function for minimization
        # We want to maximize d_min.
        # So we minimize -d_min.
        # To make it smooth, we can use a soft min, but L-BFGS-B can handle non-smooth to some extent?
        # Better: minimize the sum of squares of (d_min - d_ij) for d_ij < d_min? No.
        # Let's just use the actual min distance. It's non-smooth but often works with Nelder-Mead or Powell.
        # Or, use a barrier: minimize -log(d_min) ? No.
        
        # Let's use a penalty method on d_max.
        # Maximize d_min subject to d_max <= 1.
        # This is hard.
        
        # Alternative: Maximize d_min. The scale is free.
        # If we maximize d_min, the points will spread out infinitely.
        # So we must fix the scale.
        # Let's fix d_max = 1.
        # Then we maximize d_min subject to d_max <= 1.
        
        def objective(params):
            points = params.reshape(n, 2)
            d = pdist(points)
            d_max = np.max(d)
            d_min = np.min(d)
            # Penalty if d_max > 1
            penalty = 0
            if d_max > 1:
                penalty = (d_max - 1) ** 2 * 1000
            return -d_min + penalty
            
        # Initializations
        best_points = None
        best_score = -1
        
        # Try different ring configurations
        splits = [(6, 9), (5, 10), (7, 8), (4, 11), (8, 7), (10, 5), (9, 6)]
        
        for n1, n2 in splits:
            # Create initial points
            # Center
            p_center = np.array([0.0, 0.0])
            
            # Inner ring
            r1 = 0.5
            angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
            p_inner = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
            
            # Outer ring
            r2 = 1.0
            angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False)
            # Rotate outer ring by half step to interleave?
            angles2 += np.pi / n2
            p_outer = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
            
            points_init = np.vstack([p_center, p_inner, p_outer])
            
            # Optimize
            # Use Powell or Nelder-Mead as they don't require gradients and handle non-smoothness better
            res = minimize(objective, points_init.flatten(), method='Powell', options={'maxiter': 1000, 'maxfev': 10000})
            
            points_opt = res.x.reshape(n, 2)
            s = score(points_opt)
            
            if s > best_score:
                best_score = s
                best_points = points_opt
                
        return best_points
