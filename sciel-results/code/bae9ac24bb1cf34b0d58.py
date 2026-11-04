import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import itertools

N = 13

# EVOLVE-BLOCK-START
def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns negative score for minimization.
    Vectorized calculation of triangle areas.
    """
    points = np.asarray(points)
    
    # 1. Calculate Convex Hull Area
    try:
        # Check for degeneracy (all points collinear or identical)
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    # 2. Calculate Minimum Triangle Area (Vectorized)
    # Get all unique pairs of indices for triangles
    # Instead of itertools, we can use meshgrid or just loop if vectorized is too complex to write inline.
    # For N=13, 286 triangles is small. Let's stick to a simple loop but optimized, or vectorize.
    # Vectorization:
    # i, j, k are indices.
    # We can create arrays of all combinations.
    idx = np.arange(N)
    i, j, k = np.array(list(itertools.combinations(idx, 3))).T
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Cross product in 2D: v1[0]*v2[1] - v1[1]*v2[0]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 1e-12:
        return 0.0
        
    score = min_area / hull_area
    return -score

def construct():
    # Use Differential Evolution for global optimization
    # Bounds: each coordinate in [-1, 1]
    bounds = [(-1, 1)] * (2 * N)
    
    # Differential Evolution parameters
    # popsize: 15 * dimension? 15 * 26 = 390. Might be slow. Let's try 10.
    # maxiter: 100
    # tol: 1e-4
    # update_interval: 1
    # seed: 42
    
    # To ensure it finishes within 120s, we need to be careful with popsize and maxiter.
    # 26 dimensions.
    # Let's use a smaller popsize and fewer iterations, but it's a global search.
    # Alternatively, we can use a hybrid: DE for a short time, then Nelder-Mead for polishing.
    # But the directive says "new program ... from scratch". DE alone is a distinct approach.
    
    # Let's try popsize=10, maxiter=50.
    # Each function evaluation is fast (vectorized).
    # 10 * 26 * 50 = 13000 evaluations? No, popsize is the number of individuals in the population.
    # Total evaluations approx = popsize * maxiter * (some factor).
    # Let's be conservative.
    
    try:
        res = differential_evolution(_score, bounds, 
                                     maxiter=100, 
                                     popsize=10, 
                                     tol=1e-4,
                                     seed=42,
                                     init='latinhypercube',
                                     polish=False) # Disable polish to save time, we can add it if needed
        opt_points = res.x.reshape(-1, 2)
        
        # Optional: Quick local polish with Nelder-Mead on the best result from DE
        # This combines the robustness of DE with the precision of NM.
        # from scipy.optimize import minimize
        # res_nm = minimize(_score, opt_points.flatten(), method='Nelder-Mead', options={'maxiter': 2000})
        # opt_points = res_nm.x.reshape(-1, 2)
        
    except Exception:
        # Fallback to regular polygon if DE fails
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        opt_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return opt_points
# EVOLVE-BLOCK-END
