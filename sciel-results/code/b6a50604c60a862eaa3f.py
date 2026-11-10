# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
# EVOLVE-BLOCK-END

# Precompute sofa points once at module level for speed
_SOFA_U = np.arange(-3.0, 1.5, 0.01)
_SOFA_V = np.arange(-1.5, 1.5, 0.01)
_SOFA_X, _SOFA_Y = np.meshgrid(_SOFA_U, _SOFA_V, indexing='ij')
_SOFA_PTS = np.stack([_SOFA_X.ravel(), _SOFA_Y.ravel()], axis=1)  # shape (N, 2)
_N_SOFA = len(_SOFA_PTS)

def _generate_placements(waypoints):
    """
    Generate the full list of placements (including start (0,0,0) and end (-2,0,90)).
    Each placement is (x, y, a_degrees).
    """
    P0 = np.array([0.0, 0.0, 0.0])
    PN = np.array([-2.0, 0.0, 90.0])
    full = [P0] + [np.array(p, dtype=np.float64) for p in waypoints] + [PN]
    placements = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da - 1e-9)))
        for k in range(1, m + 1):
            t = k / m
            placements.append((1 - t) * p1 + t * p2)
    return np.array(placements, dtype=np.float64)

def _compute_area(waypoints):
    """
    Compute the area of the sofa (intersection of all corridor placements).
    """
    placements = _generate_placements(waypoints)
    total = len(placements)
    if total == 0:
        return 0.0
    
    mask = np.ones(_N_SOFA, dtype=bool)
    for p in placements:
        xx, yy, aa = p[0], p[1], np.radians(p[2])
        ca, sa = np.cos(aa), np.sin(aa)
        dx = _SOFA_PTS[:, 0] - xx
        dy = _SOFA_PTS[:, 1] - yy
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        inside = ((u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)) | \
                 ((u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0))
        mask &= inside
        if not mask.any():
            return 0.0  # Early exit if no points remain
    area = mask.sum() * 0.0001
    return area

def _compute_smooth_value(waypoints, beta=10.0):
    """
    Compute a smooth surrogate for the area.
    For each sofa point, calculate the minimum "distance" to the boundary of the corridor
    across all placements. If a point is inside all, distance is positive.
    We use a soft-min of the negative distances to create a smooth objective.
    Actually, simpler: Sum of exp(-beta * margin) for each point, where margin is min clearance.
    If margin > 0, exp is small (good). If margin < 0, exp is large (bad).
    We want to MINIMIZE this sum.
    """
    placements = _generate_placements(waypoints)
    total_points = _N_SOFA
    
    # Initialize margins to infinity
    min_margins = np.full(total_points, np.inf)
    
    for p in placements:
        xx, yy, aa = p[0], p[1], np.radians(p[2])
        ca, sa = np.cos(aa), np.sin(aa)
        dx = _SOFA_PTS[:, 0] - xx
        dy = _SOFA_PTS[:, 1] - yy
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        
        # Distance to boundary of the L-shape in (u,v) coords
        # L-shape: Union of R1: [0,1]x[-3,1] and R2: [-3,1]x[0,1]
        # A point is inside if it is in R1 or R2.
        # Margin is min(dist_to_R1_boundary, dist_to_R2_boundary) if inside?
        # No, margin is the distance to the boundary of the union.
        
        # Let's compute signed distance to the union.
        # dist to R1: 
        # du1 = max(0, -u, u-1)
        # dv1 = max(0, -v-3, v-1) -> wait, v in [-3, 1]. So lower -3, upper 1.
        # dv1 = max(0, v - 1, -3 - v)
        # dist_R1 = sqrt(du1^2 + dv1^2)
        
        # dist to R2:
        # du2 = max(0, -u-3, u-1) -> u in [-3, 1]. Lower -3, upper 1.
        # du2 = max(0, u - 1, -3 - u)
        # dv2 = max(0, -v, v-1) -> v in [0, 1].
        
        du1 = np.maximum(0.0, np.maximum(-u, u - 1.0))
        dv1 = np.maximum(0.0, np.maximum(-3.0 - v, v - 1.0))
        dist_R1 = np.sqrt(du1**2 + dv1**2)
        
        du2 = np.maximum(0.0, np.maximum(-3.0 - u, u - 1.0))
        dv2 = np.maximum(0.0, np.maximum(-v, v - 1.0))
        dist_R2 = np.sqrt(du2**2 + dv2**2)
        
        # Inside R1 if dist_R1 == 0? No, dist is 0 inside.
        # But we need signed distance.
        # If inside R1, dist is 0. If outside, dist > 0.
        # This gives distance to the SET, not signed distance to the BOUNDARY of the union.
        # If a point is in R1 but not R2, dist_R1=0, dist_R2>0. Union dist = 0.
        # If a point is outside both, dist_R1>0, dist_R2>0. Union dist = min(dist_R1, dist_R2).
        # This is exactly the distance to the union!
        
        dist_union = np.minimum(dist_R1, dist_R2)
        
        # We want the point to be INSIDE.
        # If inside, dist_union = 0.
        # If outside, dist_union > 0.
        # So we want to minimize the distance to the union?
        # Wait, if I am inside, distance is 0. If I am outside, distance is positive.
        # So minimizing the sum of distances would push points to the boundary or inside.
        # But we need them to be inside ALL placements.
        # So for each point, we need min over placements of ( - dist_union ) to be >= 0?
        # No. We need dist_union == 0 for all placements.
        # So we need max over placements of dist_union to be 0.
        # So we want to minimize the max distance.
        # A smooth approximation of max is soft-max.
        # Or simply sum of exp(beta * dist_union).
        
        # Let's track the max distance for each point across all placements.
        # Actually, we can just accumulate the penalty.
        # Penalty for this placement = exp(beta * dist_union)
        # Total Penalty = Sum over points of (Max over placements of exp(beta * dist_union)) ?
        # No, the sofa is the INTERSECTION.
        # Point is in sofa if it is in ALL placements.
        # So for each point, if it is outside ANY placement, it is not in sofa.
        # So we want to minimize the number of points that are outside at least one placement.
        # Smooth objective: Sum over points of max(0, max_over_placements(dist_union))
        # Even better: Sum over points of exp(beta * max_over_placements(dist_union))
        
        # To compute this efficiently:
        # For each point, we need the MAX distance to the union across all placements.
        # Let's store the max distance for each point so far.
        
        # Current max distance for each point:
        # If dist_union > current_max, update.
        
        # However, computing max over all placements for all points is O(N_placements * N_points).
        # This is what _compute_area does with masks.
        # Let's try to be smart.
        
        # We can just compute the penalty for this placement and sum it up?
        # No, that would be Union of placements (OR logic).
        # We need Intersection (AND logic).
        # AND logic in minimization: Minimize max(costs).
        
        # So we need to keep track of the worst-case (max distance) for each point.
        
        # Initialize max_dist for each point to 0 (or -inf if we consider inside as 0).
        # Actually, if inside, dist=0. If outside, dist>0.
        # So we want to minimize max(dist).
        
        # Let's maintain an array `worst_dist` of size N_SOFA, initialized to 0.
        # For each placement:
        #   dist_union = ...
        #   worst_dist = np.maximum(worst_dist, dist_union)
        
        # Then objective = sum(exp(beta * worst_dist))
        
        # This is O(N_placements * N_points) which is acceptable.
        
        # Let's refactor to do this outside the loop to avoid allocating every time?
        # No, we are inside the loop for each placement.
        
        # Let's do it.
        
        # Wait, I already computed dist_union.
        # I need to update a global max array.
        
        # I will restructure _compute_smooth_value to maintain the max array.
        
        pass # Placeholder

    # This approach requires maintaining state across the placement loop.
    # Let's rewrite _compute_smooth_value properly.
    
    # Re-implementing properly:
    # Initialize worst_dist
    worst_dist = np.zeros(total_points)
    
    for p in placements:
        xx, yy, aa = p[0], p[1], np.radians(p[2])
        ca, sa = np.cos(aa), np.sin(aa)
        dx = _SOFA_PTS[:, 0] - xx
        dy = _SOFA_PTS[:, 1] - yy
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        
        du1 = np.maximum(0.0, np.maximum(-u, u - 1.0))
        dv1 = np.maximum(0.0, np.maximum(-3.0 - v, v - 1.0))
        dist_R1 = np.sqrt(du1**2 + dv1**2)
        
        du2 = np.maximum(0.0, np.maximum(-3.0 - u, u - 1.0))
        dv2 = np.maximum(0.0, np.maximum(-v, v - 1.0))
        dist_R2 = np.sqrt(du2**2 + dv2**2)
        
        dist_union = np.minimum(dist_R1, dist_R2)
        
        # Update worst-case distance
        worst_dist = np.maximum(worst_dist, dist_union)
        
        # Optional early exit? If worst_dist is huge, it won't get better.
        # But we need the sum.
        
    # Objective: Sum of exp(beta * worst_dist)
    # If worst_dist is 0 for all, sum is N.
    # If some are outside, sum increases.
    # We want to MINIMIZE this.
    
    # To prevent overflow, clip worst_dist?
    # Or just use a moderate beta.
    
    val = np.sum(np.exp(beta * worst_dist))
    return val

def _neg_smooth_area(params, beta=10.0):
    """
    Objective function for minimization.
    params: flat array of waypoints (3 * N_waypoints)
    """
    N_waypoints = len(params) // 3
    waypoints = params.reshape(N_waypoints, 3)
    
    # Ensure angles are reasonable?
    # Nelder-Mead might explore crazy angles.
    # Clip angles to [-180, 180] or similar?
    # The problem allows any angle, but physically 0-90 is the turn.
    # Let's clip to [-90, 180] to avoid spinning.
    waypoints[:, 2] = np.clip(waypoints[:, 2], -90, 180)
    
    val = _compute_smooth_value(waypoints, beta=beta)
    return val

def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of waypoints to maximize the sofa area.
    Uses Nelder-Mead on a smooth surrogate of the area.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_deadline = deadline - 10.0 # Leave 10s for final area calc and return

    # Define number of waypoints
    # More waypoints = more flexibility but harder to optimize.
    # 10 waypoints seems reasonable.
    N_waypoints = 10
    
    # Initial guess:
    # A reasonable path:
    # Start (0,0,0). End (-2,0,90).
    # Waypoints should guide the corner.
    
    # Let's create a grid of initial guesses or a heuristic start.
    # Heuristic:
    # x: 0 to -2
    # y: 0 to 1 to 0
    # a: 0 to 90
    
    # Parametric initial guess:
    t = np.linspace(0, 1, N_waypoints)
    x_init = -2 * t
    y_init = np.sin(np.pi * t) # Arch up
    a_init = 90 * t
    
    initial_params = np.column_stack([x_init, y_init, a_init]).ravel()
    
    # Nelder-Mead options
    options = {
        'maxiter': 1000,
        'xatol': 1e-4,
        'fatol': 1e-6,
        'disp': False
    }
    
    best_result = None
    best_smooth_val = np.inf
    best_params = initial_params
    
    # Run Nelder-Mead
    # We can try a few different initial guesses if time permits.
    
    # For now, just one run to see if it works.
    
    # We need to check time inside the objective?
    # Nelder-Mead doesn't support callback easily for stopping.
    # We can wrap the objective to check time.
    
    def time_checked_obj(params):
        if time.time() > safety_deadline:
            raise Exception("Time limit exceeded")
        return _neg_smooth_area(params)
        
    try:
        res = minimize(time_checked_obj, initial_params, method='Nelder-Mead', options=options)
        if res.success or res.fun < best_smooth_val:
            best_smooth_val = res.fun
            best_params = res.x
            best_result = res
    except Exception as e:
        # If time limit or other error, use current best
        pass
        
    # If we have time, try a second run from a different start?
    # Or just refine the best_params?
    
    # Let's try to run a second Nelder-Mead from the best_params found, with tighter tolerance.
    if time.time() < safety_deadline - 5:
        try:
            options2 = {
                'maxiter': 500,
                'xatol': 1e-5,
                'fatol': 1e-7,
                'disp': False
            }
            res2 = minimize(time_checked_obj, best_params, method='Nelder-Mead', options=options2)
            if res2.fun < best_smooth_val:
                best_smooth_val = res2.fun
                best_params = res2.x
        except Exception:
            pass
            
    # Convert best_params to waypoints
    waypoints = best_params.reshape(N_waypoints, 3)
    
    # Validate
    if not np.all(np.isfinite(waypoints)):
        # Fallback
        t = np.linspace(0, 1, N_waypoints)
        waypoints = np.column_stack([-2*t, np.sin(np.pi*t), 90*t])
        
    # Calculate actual area for final check (optional, but good for logging/debug)
    # area = _compute_area(waypoints)
    
    return waypoints.astype(np.float64)

# EVOLVE-BLOCK-START
# (code already defined above)
# EVOLVE-BLOCK-END
