import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import time

N = 13

# EVOLVE-BLOCK-START
def construct():
    n = N
    
    # Precompute triangle indices for vectorized area calculation
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    
    def score_points(pts):
        """
        Calculate the score: min triangle area / convex hull area.
        Returns negative score for minimization.
        """
        try:
            # Check for valid hull
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 1000.0 # Penalty for degenerate hulls
        except:
            return 1000.0
            
        # Vectorized triangle area calculation
        p1 = pts[tri_indices[:, 0]]
        p2 = pts[tri_indices[:, 1]]
        p3 = pts[tri_indices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        score = min_area / hull_area
        return -score

    def objective(x):
        """
        Objective function for differential_evolution.
        x is a flat array of 2*n coordinates.
        """
        pts = x.reshape((n, 2))
        return score_points(pts)

    # Define bounds for the search space
    # Since the score is scale-invariant, we can bound the coordinates
    # to a reasonable range. [-2, 2] is sufficient.
    bounds = [(-2.0, 2.0)] * (2 * n)
    
    # Run Differential Evolution
    # maxiter: maximum number of iterations
    # popsize: size of the population (multiplied by number of parameters)
    # tol: tolerance for termination
    # seed: for reproducibility, though not strictly required, helps in debugging
    start_time = time.time()
    
    # We have ~120s. DE is fast. 
    # Let's try a moderately high popsize and maxiter.
    # 26 params. popsize=15 -> 390 individuals per generation.
    # 100 iterations -> 39000 evaluations.
    # Each eval is ~286 triangle calcs + 1 hull calc. Very fast in NumPy.
    # This should run in seconds.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=20, 
            tol=1e-6, 
            seed=42,
            polish=False # Nelder-Mead polish can be slow and sometimes get stuck; DE itself is usually good enough
        )
    except Exception as e:
        # Fallback to a known good configuration if DE fails
        i = np.arange(n)
        r = np.sqrt(i / n)
        theta = i * 2.399963229728653
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))

    best_points = result.x.reshape((n, 2))
    
    # Optional: A few steps of local refinement (Nelder-Mead) to squeeze out more?
    # The prompt asks for a new approach. DE is the core. 
    # Adding a local polish step is standard practice and improves results significantly.
    # Let's add a quick Nelder-Mead polish if time permits.
    
    if time.time() - start_time < 100:
        from scipy.optimize import minimize
        try:
            # Local refinement
            res_local = minimize(
                objective, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_local.fun < objective(best_points.flatten()):
                best_points = res_local.x.reshape((n, 2))
        except:
            pass

    return best_points
# EVOLVE-BLOCK-END
