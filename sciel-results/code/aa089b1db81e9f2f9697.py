import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

def calc_ratio(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        # Check for degenerate points
        if len(points) < 3:
            return 0.0
            
        # Calculate convex hull area
        hull = scipy.spatial.ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        
        if hull_area < 1e-10:
            return 0.0
            
        # Calculate all triangle areas
        # Number of triangles is N*(N-1)*(N-2)/6
        # For N=14, this is 364.
        # Vectorized calculation
        
        # Indices for i < j < k
        # Using np.indices is efficient for small N
        idx = np.indices((N, N, N))
        mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
        i, j, k = idx[0][mask], idx[1][mask], idx[2][mask]
        
        pi = points[i]
        pj = points[j]
        pk = points[k]
        
        # Cross product of (pj - pi) and (pk - pi)
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        
        areas = 0.5 * np.abs(cross)
        
        if len(areas) == 0:
            return 0.0
            
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def objective(points_flat):
    """
    Objective function for minimization.
    Minimizes negative score.
    """
    points = points_flat.reshape(-1, 2)
    score = calc_ratio(points)
    return -score

def normalize_points(points):
    """
    Scale points so that the convex hull area is 1.0.
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
        area = hull.volume
        if area < 1e-10:
            return points
        scale_factor = 1.0 / np.sqrt(area)
        return points * scale_factor
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Regular Polygon Initialization
    # Points on a circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 1.0
    init_points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    
    # List of initial configurations to try
    inits = []
    
    # 1. Regular polygon
    inits.append(init_points.copy())
    
    # 2. Perturbed regular polygons
    rng = np.random.RandomState(42)
    for _ in range(5):
        noise = rng.uniform(-0.1, 0.1, (N, 2))
        perturbed = init_points + noise
        # Ensure they don't collapse
        inits.append(perturbed)
        
    # 3. Random points in unit disk (might find better local optima?)
    # Actually, random points in disk usually have smaller min triangle area.
    # But it's worth a shot if time permits.
    for _ in range(3):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        random_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(random_pts)

    # Optimization Loop
    for idx_init, current_init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Normalize initial points to unit area to keep scale consistent
        current_init = normalize_points(current_init)
        
        # Try Powell first, then Nelder-Mead
        methods = ['Powell', 'Nelder-Mead']
        
        for method in methods:
            if time.time() > deadline:
                break
                
            try:
                # Determine maxiter based on time remaining
                time_left = deadline - time.time()
                # Rough estimate: 1000 iterations per 10 seconds?
                # Powell can be slow. Nelder-Mead is faster per iter but might need more.
                maxiter = int(max(100, time_left * 20))
                
                # Options for Powell
                if method == 'Powell':
                    options = {'maxiter': maxiter, 'ftol': 1e-9, 'xtol': 1e-9}
                else:
                    options = {'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-9, 'adaptive': True}
                
                res = scipy.optimize.minimize(
                    objective, 
                    current_init.flatten(), 
                    method=method, 
                    options=options
                )
                
                optimized_points = res.x.reshape(-1, 2)
                
                # Normalize to unit area
                optimized_points = normalize_points(optimized_points)
                
                score = calc_ratio(optimized_points)
                
                if score > best_score:
                    best_score = score
                    best_points = optimized_points.copy()
                    
            except Exception:
                pass

    # If no points found or score is 0, return a regular polygon as fallback
    if best_points is None:
        best_points = normalize_points(init_points)
    else:
        # Final normalization to ensure strict unit area hull
        best_points = normalize_points(best_points)
        
    return best_points
