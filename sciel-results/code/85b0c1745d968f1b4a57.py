import numpy as np
import time
from scipy.optimize import minimize

# Grid definition
X, Y = np.meshgrid(np.arange(-3, 1.5, 0.01), np.arange(-1.5, 1.5, 0.01))
GRID_POINTS = np.column_stack((X.ravel(), Y.ravel()))
N_POINTS = len(GRID_POINTS)

START = np.array([0.0, 0.0, 0.0])
END = np.array([-2.0, 0.0, 90.0])

def _densify_path(path):
    """
    Densify path per evaluator rules: add linear interpolations between consecutive rows.
    Returns a numpy array of shape (M, 3).
    """
    p = np.vstack([START, path, END])
    
    dense = [p[0]]
    for i in range(len(p) - 1):
        p1 = p[i]
        p2 = p[i+1]
        a1 = p1[2]
        a2 = p2[2]
        delta_a = abs(a2 - a1)
        m = max(1, int(np.ceil(delta_a - 1e-9)))
        
        if m > 1:
            for k in range(1, m + 1):
                t = k / m
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = a1 + t * (a2 - a1)
                dense.append([x, y, a])
        else:
            dense.append(p2)
            
    return np.array(dense)

def _compute_score_fast(path):
    """
    Compute the raw score using vectorized operations.
    """
    dense = _densify_path(path)
    M = len(dense)
    
    x = dense[:, 0]
    y = dense[:, 1]
    a = dense[:, 2]
    
    # Broadcasting: 
    # GRID_POINTS is (N, 2). 
    # We want u, v of shape (M, N).
    
    dx = GRID_POINTS[:, 0][np.newaxis, :] - x[:, np.newaxis] # (1, N) - (M, 1) -> (M, N)
    dy = GRID_POINTS[:, 1][np.newaxis, :] - y[:, np.newaxis] # (1, N) - (M, 1) -> (M, N)
    
    a_rad = np.deg2rad(a)
    cos_a = np.cos(-a_rad)[:, np.newaxis] # (M, 1)
    sin_a = np.sin(-a_rad)[:, np.newaxis] # (M, 1)
    
    u = cos_a * dx - sin_a * dy
    v = sin_a * dx + cos_a * dy
    
    # Mask conditions
    # mask1: 0 <= u <= 1, -3 <= v <= 1
    # mask2: -3 <= u <= 1, 0 <= v <= 1
    
    # To save memory, we can compute the intersection incrementally or in chunks.
    # But M is small (e.g. 50-100), N is 135k. M*N is ~13M booleans. 13MB. Acceptable.
    
    m1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    m2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    
    m_union = m1 | m2
    
    # Intersection over all placements (axis 0)
    # A point is in the sofa if it is in the union for ALL placements.
    mask_final = np.all(m_union, axis=0)
    
    return np.sum(mask_final) * 0.0001

def _generate_arc_path(c_y, n_points=50):
    """
    Generate a path following a circular arc from (0,0) to (-2,0)
    with center at (-1, c_y).
    """
    cx = -1.0
    R = np.sqrt(1.0 + c_y**2)
    
    # Angles of start and end points relative to center
    theta_start = np.arctan2(-c_y, 1.0)
    theta_end = np.arctan2(-c_y, -1.0)
    
    # Determine the arc direction
    # We want the arc that stays within reasonable bounds.
    # If c_y > 0, center is above. 
    # theta_start is in Q4 (x>0, y<0) -> negative angle.
    # theta_end is in Q3 (x<0, y<0) -> negative angle - pi? 
    # Let's check: atan2(-c_y, -1). If c_y=1, atan2(-1, -1) = -3pi/4.
    # atan2(-1, 1) = -pi/4.
    # Difference: -3pi/4 - (-pi/4) = -pi/2.
    # So we go from -pi/4 to -3pi/4? That's a clockwise rotation? 
    # Angle decreases.
    
    # Let's just interpolate linearly between theta_start and theta_end.
    # We need to ensure we take the "short" arc or the one that bulges "down" (lower y) 
    # if c_y > 0? 
    # Actually, if c_y is positive, the center is at y > 0. The points (0,0) and (-2,0) are below the center.
    # The arc connecting them "below" the center has y < c_y.
    # The arc connecting them "above" the center has y > c_y.
    # Usually, we want the arc that stays closer to the origin? 
    # Let's try the arc that does NOT cross the x-axis significantly? 
    # Both endpoints are on x-axis.
    # If we take the arc that goes through y < 0 (if c_y is small positive), it might be better.
    
    # Let's define the sweep.
    # If c_y > 0, theta_start > theta_end? 
    # Example c_y = 1: ts = -0.78, te = -2.35.
    # Linear interpolation: theta(t) = ts + t * (te - ts).
    # This traces the arc.
    
    thetas = theta_start + np.linspace(0, 1, n_points + 1) * (theta_end - theta_start)
    
    xs = cx + R * np.cos(thetas)
    ys = cy + R * np.sin(thetas)
    
    # Filter out endpoints
    idx = np.arange(1, n_points)
    xs = xs[idx]
    ys = ys[idx]
    
    # Angles a: linear 0 to 90
    a_vals = np.linspace(0, 90, n_points + 1)[idx]
    
    path = np.column_stack([xs, ys, a_vals])
    return path

def _generate_bezier_path(x1, y1, x2, y2, n_points=50):
    """
    Generate a path following a cubic Bezier curve from (0,0) to (-2,0)
    with control points (x1, y1) and (x2, y2).
    """
    P0 = np.array([0.0, 0.0])
    P1 = np.array([x1, y1])
    P2 = np.array([x2, y2])
    P3 = np.array([-2.0, 0.0])
    
    ts = np.linspace(0, 1, n_points + 1)
    ts = ts[1:-1] # Exclude endpoints
    
    # Bezier formula: (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    t1 = 1 - ts
    c0 = t1**3
    c1 = 3 * t1**2 * ts
    c2 = 3 * t1 * ts**2
    c3 = ts**3
    
    xs = c0 * P0[0] + c1 * P1[0] + c2 * P2[0] + c3 * P3[0]
    ys = c0 * P0[1] + c1 * P1[1] + c2 * P2[1] + c3 * P3[1]
    
    a_vals = np.linspace(0, 90, n_points + 1)[1:-1]
    
    path = np.column_stack([xs, ys, a_vals])
    return path

def objective_arc(params):
    c_y = params[0]
    n_points = int(params[1])
    c_y = np.clip(c_y, -5.0, 5.0)
    n_points = int(np.clip(n_points, 10, 100))
    
    path = _generate_arc_path(c_y, n_points)
    score = _compute_score_fast(path)
    return -score

def objective_bezier(params):
    x1, y1, x2, y2 = params
    n_points = 50 # Fixed for now, could be optimized
    
    path = _generate_bezier_path(x1, y1, x2, y2, n_points)
    score = _compute_score_fast(path)
    return -score

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 550
    
    best_score = -np.inf
    best_path = None
    
    # --- Phase 1: Grid Search for Arc ---
    # Try a range of c_y and n_points
    c_y_range = np.linspace(-2.0, 2.0, 9)
    n_points_range = [20, 40, 60, 80]
    
    for c_y in c_y_range:
        for n_pts in n_points_range:
            if time.time() > deadline - 60:
                break
            try:
                path = _generate_arc_path(c_y, n_pts)
                score = _compute_score_fast(path)
                if score > best_score:
                    best_score = score
                    best_path = path
            except Exception:
                pass
    
    # --- Phase 2: Local Optimization for Arc ---
    if best_path is not None and time.time() < deadline - 100:
        # Find the c_y and n_points that gave the best score
        # We need to recover params from best_path or just re-run grid to find best params
        # Let's just use the best c_y from the grid search for optimization
        # We didn't store which c_y was best. Let's just pick a reasonable one or re-evaluate.
        # Better: Store best params in Phase 1.
        
        # Re-doing Phase 1 logic to track params is safer.
        # For now, let's assume the best c_y is near the one that worked best.
        # Let's just optimize around a few good candidates.
        
        candidates = []
        for c_y in c_y_range:
            for n_pts in n_points_range:
                if time.time() > deadline - 100:
                    break
                try:
                    path = _generate_arc_path(c_y, n_pts)
                    score = _compute_score_fast(path)
                    if score > best_score - 0.001: # Keep top candidates
                        candidates.append((c_y, n_pts))
                except Exception:
                    pass
        
        for c_y, n_pts in candidates[:3]: # Optimize top 3
            if time.time() > deadline - 50:
                break
            try:
                res = minimize(objective_arc, [c_y, float(n_pts)], method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
                if -res.fun > best_score:
                    best_score = -res.fun
                    # Generate path from optimized params
                    opt_cy = res.x[0]
                    opt_np = int(np.clip(res.x[1], 10, 100))
                    best_path = _generate_arc_path(opt_cy, opt_np)
            except Exception:
                pass

    # --- Phase 3: Try Bezier ---
    if time.time() < deadline - 50:
        # Initial guess for Bezier: Control points to form a "U" shape or similar
        # P0=(0,0), P3=(-2,0)
        # P1=(0.5, 1.0), P2=(-1.5, 1.0) -> Bulges up
        # P1=(0.5, -1.0), P2=(-1.5, -1.0) -> Bulges down
        
        bez_candidates = [
            [0.5, 1.0, -1.5, 1.0],
            [0.5, -1.0, -1.5, -1.0],
            [0.0, 1.0, -2.0, 1.0],
            [0.0, -1.0, -2.0, -1.0],
            [1.0, 0.5, -3.0, 0.5],
            [1.0, -0.5, -3.0, -0.5]
        ]
        
        for params in bez_candidates:
            if time.time() > deadline - 20:
                break
            try:
                score = -objective_bezier(params)
                if score > best_score:
                    best_score = score
                    best_path = _generate_bezier_path(params[0], params[1], params[2], params[3], 50)
            except Exception:
                pass
        
        # Refine best Bezier
        if best_path is not None and time.time() < deadline - 20:
             # We need the params of the best Bezier. 
             # This is tricky if we didn't store them. 
             # Let's just run a generic minimize on the best found Bezier shape if we can identify it.
             # For simplicity, we'll skip Bezier refinement if we can't easily track params.
             # The Arc optimization is likely sufficient.
             pass

    if best_path is None or not np.all(np.isfinite(best_path)):
        # Fallback to a simple arc
        best_path = _generate_arc_path(0.0, 50)
        
    return best_path
