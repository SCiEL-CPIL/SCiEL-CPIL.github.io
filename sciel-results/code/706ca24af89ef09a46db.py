import numpy as np
import time
import itertools
import scipy.optimize as opt
import scipy.spatial as spat

N = 13
DEADLINE = 150.0 # Seconds
START_TIME = time.time()

# Precompute triangle indices
TRI_INDICES = list(itertools.combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute minimum triangle area
    # Vectorized calculation for all 286 triangles
    # points shape: (N, 2)
    # indices shape: (M, 3)
    p1 = points[TRI_INDICES[:, 0]]
    p2 = points[TRI_INDICES[:, 1]]
    p3 = points[TRI_INDICES[:, 2]]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def objective_flat(x):
    """
    Objective function for the optimizer.
    We want to MAXIMIZE calculate_score, so we MINIMIZE -calculate_score.
    """
    pts = x.reshape(N, 2)
    score = calculate_score(pts)
    return -score

def normalize_hull_area(points):
    """
    Scale points so that the convex hull area is 1.
    This helps in keeping the problem well-scaled.
    """
    try:
        hull = spat.ConvexHull(points)
        area = hull.volume
        if area > 1e-9:
            scale = 1.0 / np.sqrt(area)
            return points * scale
    except Exception:
        pass
    return points

def generate_initial_configurations():
    """
    Generate several symmetric initial configurations.
    """
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = np.ones(N)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    configs.append(normalize_hull_area(pts))
    
    # 2. 1 center + 12 ring
    pts = np.zeros((N, 2))
    pts[0] = [0, 0]
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    radii = np.ones(12) * 2.0 # Radius adjusted to keep hull area similar? 
    # Let's just place them and normalize later
    pts[1:] = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    configs.append(normalize_hull_area(pts))
    
    # 3. 1 center + 3 inner + 4 mid + 5 outer
    pts = np.zeros((N, 2))
    idx = 0
    pts[idx] = [0, 0]; idx += 1
    for r, count in [(0.5, 3), (1.0, 4), (1.5, 5)]:
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        # Offset angles for better distribution
        offset = 0.5 * 2*np.pi / count if count > 2 else 0
        pts[idx:idx+count] = np.column_stack([r * np.cos(angles + offset), r * np.sin(angles + offset)])
        idx += count
    configs.append(normalize_hull_area(pts))
    
    # 4. 1 center + 4 inner + 4 mid + 4 outer (13 = 1+4+4+4)
    pts = np.zeros((N, 2))
    idx = 0
    pts[idx] = [0, 0]; idx += 1
    for r, count in [(0.5, 4), (1.0, 4), (1.5, 4)]:
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        offset = 0.5 * 2*np.pi / count if count > 2 else 0
        pts[idx:idx+count] = np.column_stack([r * np.cos(angles + offset), r * np.sin(angles + offset)])
        idx += count
    configs.append(normalize_hull_area(pts))
    
    return configs

def construct():
    global START_TIME
    current_time = time.time()
    elapsed = current_time - START_TIME
    time_left = DEADLINE - elapsed
    
    if time_left < 10:
        # Fallback to a simple configuration if out of time
        configs = generate_initial_configurations()
        best_pts = configs[0]
        best_score = calculate_score(best_pts)
        return best_pts

    configs = generate_initial_configurations()
    
    best_score = -np.inf
    best_pts = None
    
    # Optimize each initial configuration
    for i, init_pts in enumerate(configs):
        current_time = time.time()
        elapsed = current_time - START_TIME
        time_left = DEADLINE - elapsed
        
        if time_left < 5:
            break
            
        # Ensure we have enough time for this run. 
        # Powell method can be slow. We limit maxiter based on time.
        # Rough estimate: each function eval takes ~1ms? 286 tris.
        # Let's allow a fixed number of iterations that fits in the remaining time.
        # Or just let it run and check inside? SciPy doesn't support internal time checks easily.
        # We will use a conservative maxiter.
        
        # Strategy: Run optimization with a timeout wrapper logic is hard in scipy.
        # Instead, we run sequential optimizations and stop if time runs out.
        
        x0 = init_pts.flatten()
        
        try:
            # Use Powell method, which is robust for non-smooth functions
            # Maximize score => Minimize -score
            res = opt.minimize(
                objective_flat, 
                x0, 
                method='Powell', 
                options={'maxiter': 200, 'ftol': 1e-8, 'xtol': 1e-8}
            )
            
            if res.success or res.fun < -best_score:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_pts = res.x.reshape(N, 2)
                    
        except Exception:
            continue

        # Check time
        current_time = time.time()
        elapsed = current_time - START_TIME
        if DEADLINE - elapsed < 10:
            break

    # If we have time left, do a local refinement on the best point found
    # Using Nelder-Mead for a final polish? Or just return best.
    # Given the non-smoothness, Powell is usually sufficient.
    
    if best_pts is None:
        # Fallback
        best_pts = configs[0]
        
    return best_pts

# Keep the required function name
def construct():
    return _construct_impl()

def _construct_impl():
    return construct() # This creates a loop. Let's just rename the main logic.

# Correcting the structure to avoid recursion
def construct():
    global START_TIME
    # If START_TIME is not set, set it
    if 'START_TIME' not in globals() or START_TIME is None:
        START_TIME = time.time()
        
    current_time = time.time()
    elapsed = current_time - START_TIME
    time_left = DEADLINE - elapsed
    
    if time_left < 10:
        configs = generate_initial_configurations()
        return configs[0]

    configs = generate_initial_configurations()
    
    best_score = -np.inf
    best_pts = None
    
    for i, init_pts in enumerate(configs):
        current_time = time.time()
        elapsed = current_time - START_TIME
        time_left = DEADLINE - elapsed
        
        if time_left < 5:
            break
            
        x0 = init_pts.flatten()
        
        try:
            res = opt.minimize(
                objective_flat, 
                x0, 
                method='Powell', 
                options={'maxiter': 100, 'ftol': 1e-6, 'xtol': 1e-6}
            )
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_pts = res.x.reshape(N, 2)
                
        except Exception:
            continue

        current_time = time.time()
        elapsed = current_time - START_TIME
        if DEADLINE - elapsed < 10:
            break

    if best_pts is None:
        best_pts = configs[0]
        
    return best_pts
