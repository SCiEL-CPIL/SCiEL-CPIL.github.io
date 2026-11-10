# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _dist_to_rect(u, v, xmin, xmax, ymin, ymax):
    """
    Calculate signed distance from points (u, v) to the rectangle [xmin, xmax] x [ymin, ymax].
    Positive if inside, negative if outside.
    """
    # Distance to boundaries in x and y
    dx = np.maximum(xmin - u, 0) + np.maximum(u - xmax, 0)
    dy = np.maximum(ymin - v, 0) + np.maximum(v - ymax, 0)
    
    # If outside in both x and y, distance is hypotenuse to corner
    # If outside in one, distance is that component
    # If inside, distance is min distance to edge
    
    # Standard formula for signed distance to AABB:
    # Let dx_out = max(xmin-u, u-xmax, 0)
    # Let dy_out = max(ymin-v, v-ymax, 0)
    # dist = sqrt(dx_out^2 + dy_out^2)
    # sign = +1 if inside, -1 if outside
    
    inside = (u >= xmin) & (u <= xmax) & (v >= ymin) & (v <= ymax)
    
    # Distance to nearest face if outside, or nearest edge if inside
    # For outside:
    dx_out = np.maximum(xmin - u, u - xmax)
    dy_out = np.maximum(ymin - v, v - ymax)
    dist_out = np.sqrt(dx_out**2 + dy_out**2)
    
    # For inside:
    # Distance to nearest edge
    d_left = u - xmin
    d_right = xmax - u
    d_bottom = v - ymin
    d_top = ymax - v
    dist_in = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    
    dist = np.where(inside, dist_in, -dist_out)
    return dist

def _signed_distance_to_base(u, v):
    """
    Calculate signed distance to the base L-shaped corridor.
    Base = Union of R1 and R2.
    """
    # R1: [0, 1] x [-3, 1]
    d1 = _dist_to_rect(u, v, 0, 1, -3, 1)
    # R2: [-3, 1] x [0, 1]
    d2 = _dist_to_rect(u, v, -3, 1, 0, 1)
    
    # Distance to union is max of distances to components
    return np.maximum(d1, d2)

def _soft_sofa_score(path, k, deadline=None):
    """
    Compute soft score: sum of sigmoid(k * min_i dist_i(p))
    """
    if deadline is not None and time.time() > deadline:
        return 0.0

    # Build full placement list
    placements = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        path,
        np.array([[-2.0, 0.0, 90.0]])
    ])
    
    # Densify
    densified = [placements[0]]
    for i in range(len(placements) - 1):
        p1 = placements[i]
        p2 = placements[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        if m > 1:
            for j in range(1, m):
                t = j / m
                interp = p1 + t * (p2 - p1)
                densified.append(interp)
        densified.append(p2)
    
    densified = np.array(densified)
    n_placements = len(densified)
    
    # Sofa grid points
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sofa_points = np.column_stack([X.ravel(), Y.ravel()])
    n_points = len(sofa_points)
    
    # Initialize min distance for each point to infinity (or 0 for soft min? No, we want min)
    # We want min_i d_i(p).
    # Start with a large positive number, then update with min.
    min_dists = np.full(n_points, np.inf)
    
    for k_idx in range(n_placements):
        x, y, a = densified[k_idx]
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(a):
            continue
        a_rad = np.radians(a)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = sofa_points[:, 0] - x
        dy = sofa_points[:, 1] - y
        
        # R(-a)
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        d_base = _signed_distance_to_base(u, v)
        
        # Update min
        min_dists = np.minimum(min_dists, d_base)
        
        if deadline is not None and time.time() > deadline:
            break
            
    # If loop broke early due to deadline, min_dists might be inf for some points
    # Replace inf with a large negative number to penalize
    min_dists = np.where(np.isinf(min_dists), -100, min_dists)
    
    # Soft score: sum of sigmoid(k * min_dists)
    # sigmoid(x) = 1 / (1 + exp(-x))
    # To avoid overflow in exp, clip x
    arg = k * min_dists
    arg = np.clip(arg, -50, 50)
    sig = 1.0 / (1.0 + np.exp(-arg))
    
    return np.sum(sig)

def _compute_sofa_area(path, deadline=None):
    """
    Compute the area of the sofa (intersection of all corridor placements)
    for a given path. The evaluator adds (0,0,0) before and (-2,0,90) after.
    """
    # Build full placement list with evaluator's additions
    placements = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        path,
        np.array([[-2.0, 0.0, 90.0]])
    ])
    
    # Densify: between consecutive rows, add linear interpolations
    # in m = max(1, ceil(|a2 - a1|)) equal steps
    densified = [placements[0]]
    for i in range(len(placements) - 1):
        p1 = placements[i]
        p2 = placements[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        if m > 1:
            for j in range(1, m):
                t = j / m
                interp = p1 + t * (p2 - p1)
                densified.append(interp)
        densified.append(p2)
    
    densified = np.array(densified)
    
    # Sofa grid points
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sofa_points = np.column_stack([X.ravel(), Y.ravel()])
    n_points = len(sofa_points)
    
    # Check each placement: point is inside if R(-a)(w - (x,y)) is in base
    # R(-a) is rotation by -a degrees
    inside_all = np.ones(n_points, dtype=bool)
    
    for k in range(len(densified)):
        x, y, a = densified[k]
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(a):
            continue
        a_rad = np.radians(a)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # w - (x, y)
        dx = sofa_points[:, 0] - x
        dy = sofa_points[:, 1] - y
        
        # R(-a) * (dx, dy): rotate by -a
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        # Check if in base corridor
        inside = ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
                 ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))
        inside_all &= inside
        
        # Early termination if no points remain
        if not inside_all.any():
            break
        
        # Deadline check
        if deadline is not None and time.time() > deadline:
            break
    
    count = inside_all.sum()
    return count * 0.0001

def moving_sofa_2d() -> np.ndarray:
    """
    Optimize the path using multi-phase optimization with soft surrogate.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    # Initial path
    best_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    best_area = _compute_sofa_area(best_path, deadline)
    
    # Phase 1: Coarse Soft Optimization
    # Minimize negative soft score
    def neg_soft_score_1(params):
        path = params.reshape(8, 3)
        return -_soft_sofa_score(path, k=5, deadline=deadline)
        
    if time.time() < deadline - 100:
        res1 = minimize(neg_soft_score_1, best_path.ravel(), method='Nelder-Mead', 
                        options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if res1.success and -res1.fun > 0:
            candidate = res1.x.reshape(8, 3)
            # Validate
            if np.all(np.isfinite(candidate)):
                area = _compute_sofa_area(candidate, deadline)
                if area > best_area:
                    best_path = candidate
                    best_area = area
    
    # Phase 2: Fine Soft Optimization
    def neg_soft_score_2(params):
        path = params.reshape(8, 3)
        return -_soft_sofa_score(path, k=20, deadline=deadline)
        
    if time.time() < deadline - 50:
        res2 = minimize(neg_soft_score_2, best_path.ravel(), method='Nelder-Mead', 
                        options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
        if res2.success and -res2.fun > 0:
            candidate = res2.x.reshape(8, 3)
            if np.all(np.isfinite(candidate)):
                area = _compute_sofa_area(candidate, deadline)
                if area > best_area:
                    best_path = candidate
                    best_area = area
                    
    # Phase 3: Exact Polish with Random Walk (since exact is discontinuous)
    # Use smaller steps and more iterations
    step_size = 0.01
    n_steps = 100
    for step in range(n_steps):
        if time.time() > deadline - 10.0:
            break
            
        perturbation = np.random.randn(8, 3) * step_size
        candidate = best_path + perturbation
        
        # Keep angles monotonic-ish
        new_angles = candidate[:, 2]
        new_angles[0] = np.clip(new_angles[0], 5, 30)
        new_angles[-1] = np.clip(new_angles[-1], 70, 89)
        for i in range(1, 8):
            new_angles[i] = max(new_angles[i], new_angles[i-1] + 0.5)
            new_angles[i] = min(new_angles[i], 89.0)
        candidate[:, 2] = new_angles
        
        candidate[:, 0] = np.clip(candidate[:, 0], -1.5, 0.5)
        candidate[:, 1] = np.clip(candidate[:, 1], -0.5, 0.5)
        
        area = _compute_sofa_area(candidate, deadline)
        
        if area > best_area:
            best_path = candidate
            best_area = area
            step_size = min(step_size * 1.1, 0.05)
        else:
            step_size = max(step_size * 0.9, 0.005)

    return best_path

# EVOLVE-BLOCK-END
