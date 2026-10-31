import numpy as np
from scipy.optimize import minimize, basinhopping
import time

N = 11
SQRT3 = np.sqrt(3)

def triangle_areas_vectorized(points):
    """
    Calculate the areas of all triangles formed by the points using vectorization.
    """
    n = points.shape[0]
    # Create indices for all unique triplets (i, j, k) with i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    # Vectorized cross product magnitude
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
    return pen

def objective(flat_params):
    points = flat_params.reshape(N, 2)
    min_area = np.min(triangle_areas_vectorized(points))
    pen = penalty(points)
    return -min_area + 1000.0 * pen

def safe_objective(flat_params):
    points = flat_params.reshape(N, 2)
    if np.any(points[:, 0] < -1) or np.any(points[:, 0] > 2) or np.any(points[:, 1] < -1) or np.any(points[:, 1] > 2):
        return 1e6
    return objective(flat_params)

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Helper to get a symmetric initial guess
    # Based on known good configurations for 11 points in a triangle
    def get_symmetric_init():
        # A common symmetric arrangement involves points on the median and symmetric pairs
        # Here we construct a rough symmetric grid
        pts = []
        # Centroid
        pts.append([0.5, SQRT3/6])
        # 3 points near vertices but inside
        for i in range(3):
            angle = i * 2 * np.pi / 3 + np.pi/2 # Adjusted for orientation
            # Actually, simpler to just place them symmetrically
            pass
        
        # Let's use a specific known good starting point structure:
        # 1 center, 3 near vertices, 3 mid-edges, 3 inner ring?
        # For now, let's just use a well-spaced random symmetric set
        # 11 points: 
        # 1 at centroid
        # 3 at 1/4 height from base, spaced
        # 3 at 3/4 height from base, spaced
        # 3 near vertices?
        
        # Simpler: Generate points in barycentric coordinates that are symmetric
        # We'll just use a few random restarts and one fixed "good" guess
        return None

    # Fixed initial guess based on literature approximations or symmetry
    # 11 points: 
    # (0.5, h/3) centroid
    # (0.25, h/6), (0.75, h/6)
    # (0.25, h/2), (0.75, h/2)
    # (0.5, 2h/3)
    # (0.1, h/12), (0.9, h/12)
    # (0.1, 5h/12), (0.9, 5h/12)
    # (0.5, h/12)
    # This is just a heuristic. Let's rely on random restarts + Powell which is robust.
    
    # Try multiple random restarts with Powell
    # Increase number of restarts since objective is now faster
    for seed in range(200):
        if time.time() - start_time > time_limit - 10:
            break
        
        rng = np.random.RandomState(seed)
        init_points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            init_points[i, 0] = v * 1.0 + w * 0.5
            init_points[i, 1] = w * SQRT3 / 2.0
        
        flat_init = init_points.flatten()
        
        try:
            result = minimize(
                safe_objective, 
                flat_init, 
                method='Powell', 
                options={'maxiter': 200, 'maxfev': 20000, 'disp': False}
            )
            
            opt_points = result.x.reshape(N, 2)
            min_area = np.min(triangle_areas_vectorized(opt_points))
            pen = penalty(opt_points)
            
            if pen < 1e-4:
                score = min_area
            else:
                score = min_area - 1000 * pen
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception:
            pass

    if best_points is not None and time.time() - start_time < time_limit - 10:
        try:
            bh_result = basinhopping(
                safe_objective,
                best_points.flatten(),
                minimizer_kwargs={'method': 'Powell', 'maxiter': 100, 'maxfev': 10000},
                niter=20,
                seed=123
            )
            bh_points = bh_result.x.reshape(N, 2)
            bh_min_area = np.min(triangle_areas_vectorized(bh_points))
            bh_pen = penalty(bh_points)
            
            if bh_pen < 1e-4:
                bh_score = bh_min_area
            else:
                bh_score = bh_min_area - 1000 * bh_pen
                
            if bh_score > best_score:
                best_score = bh_score
                best_points = bh_points.copy()
        except Exception:
            pass

    if best_points is None:
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
