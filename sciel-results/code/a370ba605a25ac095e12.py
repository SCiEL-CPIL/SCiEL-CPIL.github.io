import numpy as np
import time
import math

def _build_grid():
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def _build_subgrid(step=3):
    xs = np.arange(-3.0, 1.5, 0.01 * step)
    ys = np.arange(-1.5, 1.5, 0.01 * step)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def _densify_path(path):
    """Densify path by adding linear interpolations between consecutive waypoints."""
    full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
    result = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(math.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            result.append(interp)
    return np.array(result)

def _compute_distance_to_base(grid_x, grid_y, x, y, a_rad):
    """Compute distance from each grid point to the boundary of the base corridor.
    Positive = inside, negative = outside. Returns array of distances."""
    cos_a = math.cos(a_rad)
    sin_a = math.sin(a_rad)
    
    dx = grid_x - x
    dy = grid_y - y
    u = cos_a * dx + sin_a * dy
    v = -sin_a * dx + cos_a * dy
    
    # Distance to arm1: {0 <= u <= 1, -3 <= v <= 1}
    du1 = np.maximum(0, np.maximum(0 - u, u - 1))
    dv1 = np.maximum(0, np.maximum(-3 - v, v - 1))
    dist1 = np.hypot(du1, dv1)
    
    # Distance to arm2: {-3 <= u <= 1, 0 <= v <= 1}
    du2 = np.maximum(0, np.maximum(-3 - u, u - 1))
    dv2 = np.maximum(0, np.maximum(0 - v, v - 1))
    dist2 = np.hypot(du2, dv2)
    
    # Distance to union = min of distances to each arm
    # But we need to be careful: the "inside" region is the union
    # A point is inside if it's inside arm1 OR arm2
    # Distance to boundary of union:
    # If inside arm1: distance to arm1 boundary (positive)
    # If inside arm2: distance to arm2 boundary (positive)  
    # If outside both: -min(dist1, dist2)
    
    inside1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    inside2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    
    # Signed distance: positive inside, negative outside
    # For points inside arm1 but not arm2, use dist1
    # For points inside arm2 but not arm1, use dist2
    # For points inside both, use max(dist1, dist2)
    # For points outside both, use -min(dist1, dist2)
    
    # Actually, let me think about this differently.
    # The signed distance to the union of two sets is:
    # If in union: min distance to boundary of union
    # If not in union: -min distance to union
    
    # Simpler approach: compute signed distance to each arm, then combine
    # For a convex set, signed distance is straightforward
    # For union, it's more complex
    
    # Let me use a different approach: compute the minimum of the "inside-ness"
    # Actually, for the soft-min, I need a smooth approximation of:
    # min over placements of (indicator that point is in placement)
    
    # Let me just compute the minimum margin across all placements
    pass
    return None  # Placeholder

def _evaluate_path_fast(path, grid_x, grid_y, mask):
    """Evaluate using a mask of points that are still candidates."""
    dense = _densify_path(path)
    n_points = len(grid_x)
    inside = mask.copy()
    
    for i in range(len(dense)):
        x, y, a_deg = dense[i]
        a_rad = math.radians(a_deg)
        cos_a = math.cos(a_rad)
        sin_a = math.sin(a_rad)
        
        dx = grid_x - x
        dy = grid_y - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        in_base = arm1 | arm2
        
        inside = inside & in_base
        if not np.any(inside):
            break
    
    return np.sum(inside) * 0.0001

def _soft_min_value(path, grid_x, grid_y, beta=50.0):
    """Compute a smooth surrogate for the intersection area using soft-min.
    
    For each grid point, compute the minimum "margin" across all placements.
    The soft-min approximates the hard min smoothly.
    
    Returns: (value, gradient_wrt_path_params)
    """
    dense = _densify_path(path)
    n_points = len(grid_x)
    n_placements = len(dense)
    
    # For each point, compute the margin (distance to boundary) for each placement
    # Then take soft-min over placements
    
    # Initialize: for each point, track the minimum margin
    # margin[i, j] = distance from point j to boundary of placement i
    # We want min over i of margin[i, j] for each j
    
    # To save memory, process placements one at a time
    # soft_min = -log(sum(exp(-margin/beta))) * beta
    
    log_sum = np.zeros(n_points, dtype=float)
    
    for i in range(n_placements):
        x, y, a_deg = dense[i]
        a_rad = math.radians(a_deg)
        cos_a = math.cos(a_rad)
        sin_a = math.sin(a_rad)
        
        dx = grid_x - x
        dy = grid_y - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        # Compute margin: positive if inside, negative if outside
        # For the union of two rectangles, a point is inside if in either
        # Margin = max(margin_arm1, margin_arm2) where margin is signed distance
        # Actually for union, we need: inside if in arm1 OR arm2
        # Signed distance to union:
        # If in arm1: signed_dist_arm1 (positive)
        # If in arm2: signed_dist_arm2 (positive)
        # If in both: max(signed_dist_arm1, signed_dist_arm2)
        # If in neither: -min(|dist_to_arm1|, |dist_to_arm2|)
        
        # Simpler: compute if inside and a "margin" value
        # Let me use: for each arm, compute a "slack" that's positive inside
        # slack_arm1 = min(u, 1-u, v+3, 1-v) -- positive if inside arm1
        # slack_arm2 = min(u+3, 1-u, v, 1-v) -- positive if inside arm2
        # For union: slack = max(slack_arm1, slack_arm2)
        
        slack1 = np.minimum(np.minimum(u, 1 - u), np.minimum(v + 3, 1 - v))
        slack2 = np.minimum(np.minimum(u + 3, 1 - u), np.minimum(v, 1 - v))
        
        # For the union, the "inside margin" is:
        # If in arm1: slack1
        # If in arm2: slack2
        # If in both: max(slack1, slack2)
        # If in neither: min(slack1, slack2) (both negative, so this is the less negative)
        
        # Actually, for the intersection of placements, a point is in the sofa
        # if it's inside ALL placements. The "margin" for a placement is how
        # far inside it is. The sofa margin is min over placements of margin.
        
        # For a union of two rectangles, the signed distance to the complement
        # (i.e., how far inside the union) is:
        # If in arm1: distance to boundary of arm1 (but capped by distance to arm2 boundary if closer)
        # This is getting complex. Let me use a simpler approach.
        
        # Use: margin = max(slack1, slack2)
        # This is positive if in either arm, and gives the minimum distance to
        # the boundary of whichever arm the point is in (or the closer one if in both)
        
        margin = np.maximum(slack1, slack2)
        
        # Soft-min accumulation: log_sum += exp(-margin/beta)
        # But this can overflow. Use log-sum-exp trick.
        # Actually, let's just accumulate the log-sum directly
        log_sum += (-margin / beta)
    
    # soft_min = -beta * log(sum(exp(-margin_i/beta)))
    # = -beta * log_sum (if we accumulated log(exp(...)) correctly)
    # Wait, I accumulated -margin/beta, so:
    # sum(exp(-margin_i/beta)) = sum(exp(log_terms))
    # log(sum(exp(log_terms))) = logsumexp(log_terms)
    
    # I accumulated the log_terms directly (not log of them), so:
    # log_sum = sum_i (-margin_i / beta)
    # But I need log(sum_i exp(-margin_i/beta))
    
    # Let me redo: I should accumulate exp values or use logsumexp properly
    pass
    
    # This approach is getting complicated. Let me use a simpler smooth approximation.
    # Instead of soft-min, use: for each point, compute min margin, then use
    # a smooth step function to convert to a continuous "inside" indicator.
    
    # Actually, let me just compute the hard min and use it for the value,
    # but compute gradients numerically for optimization.
    
    # For now, return the hard evaluation value and use finite differences for gradients
    return None

def _hill_climb(path, grid_x, grid_y, deadline, n_iter=2000):
    """Hill climbing optimization with adaptive step sizes."""
    n = len(path)
    best_path = path.copy()
    best_area = _evaluate_path_fast(path, grid_x, grid_y, np.ones(len(grid_x), dtype=bool))
    
    step_sizes = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    for ss in step_sizes:
        for _ in range(n_iter):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for dim in range(3):
                    for direction in [1, -1]:
                        trial = best_path.copy()
                        trial[i, dim] += direction * ss
                        if dim == 2:
                            trial[i, dim] = np.clip(trial[i, dim], -10, 100)
                        
                        area = _evaluate_path_fast(trial, grid_x, grid_y, np.ones(len(grid_x), dtype=bool))
                        if area > best_area:
                            best_path = trial
                            best_area = area
                            improved = True
            if not improved:
                break
    
    return best_path, best_area

def _gradient_descent_smooth(path, grid_x, grid_y, deadline, n_iter=500, lr=0.05, beta=20.0):
    """Gradient descent using a smooth surrogate based on soft-min.
    
    The smooth surrogate for the intersection area is:
    For each point j, compute margin_j = min_i (distance from point j to boundary of placement i)
    The sofa area ≈ sum_j (1 if margin_j > 0 else 0) * 0.0001
    
    Smooth approximation: sum_j sigmoid(beta * margin_j) * 0.0001
    
    We compute gradients numerically using finite differences on the smooth objective.
    """
    n_params = len(path) * 3
    best_path = path.copy()
    
    def smooth_objective(p):
        """Compute smooth surrogate objective."""
        p = p.reshape(-1, 3)
        dense = _densify_path(p)
        n_points = len(grid_x)
        
        # For each point, compute minimum margin across placements
        # Then apply sigmoid to get smooth "inside" indicator
        
        # Initialize min margin to +inf
        min_margin = np.full(n_points, np.inf)
        
        for i in range(len(dense)):
            x, y, a_deg = dense[i]
            a_rad = math.radians(a_deg)
            cos_a = math.cos(a_rad)
            sin_a = math.sin(a_rad)
            
            dx = grid_x - x
            dy = grid_y - y
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            # Slack for each arm (positive if inside)
            slack1 = np.minimum(np.minimum(u, 1 - u), np.minimum(v + 3, 1 - v))
            slack2 = np.minimum(np.minimum(u + 3, 1 - u), np.minimum(v, 1 - v))
            
            # Margin for union: max of the two slacks
            margin = np.maximum(slack1, slack2)
            
            min_margin = np.minimum(min_margin, margin)
        
        # Smooth indicator: sigmoid(beta * margin)
        # Clip to avoid overflow
        beta_margin = np.clip(beta * min_margin, -50, 50)
        inside_prob = 1.0 / (1.0 + np.exp(-beta_margin))
        
        return np.sum(inside_prob) * 0.0001
    
    # Compute initial objective
    current_obj = smooth_objective(best_path)
    
    # Numerical gradient
    eps = 1e-4
    for iteration in range(n_iter):
        if time.time() > deadline:
            break
        
        # Compute gradient numerically
        grad = np.zeros(n_params)
        for i in range(n_params):
            p_plus = best_path.copy()
            p_plus.ravel()[i] += eps
            if i % 3 == 2:  # angle dimension
                p_plus.ravel()[i] = np.clip(p_plus.ravel()[i], -10, 100)
            
            p_minus = best_path.copy()
            p_minus.ravel()[i] -= eps
            if i % 3 == 2:
                p_minus.ravel()[i] = np.clip(p_minus.ravel()[i], -10, 100)
            
            obj_plus = smooth_objective(p_plus)
            obj_minus = smooth_objective(p_minus)
            grad[i] = (obj_plus - obj_minus) / (2 * eps)
        
        # Gradient step
        step = lr * grad
        new_path = best_path + step.reshape(-1, 3)
        
        # Clip angles
        new_path[:, 2] = np.clip(new_path[:, 2], -10, 100)
        
        new_obj = smooth_objective(new_path)
        
        if new_obj > current_obj:
            best_path = new_path
            current_obj = new_obj
        else:
            # Reduce learning rate
            lr *= 0.8
    
    return best_path, smooth_objective(best_path)

def _ps_path(n_ctrl=8):
    """Generate the Pogossnitsky-Scheybman optimal path for the L-corridor.
    
    The analytic solution for the moving sofa problem in an L-shaped corridor.
    The path is parameterized by theta from 0 to pi/2.
    """
    theta = np.linspace(0, math.pi/2, n_ctrl)
    
    # The PS path: the sofa pivots around the inner corner
    # x = -1 + cos(theta) - sin(theta)
    # y = -1 + sin(theta) + cos(theta)
    # angle = theta in degrees
    
    x = -1.0 + np.cos(theta) - np.sin(theta)
    y = -1.0 + np.sin(theta) + np.cos(theta)
    angles = np.degrees(theta)
    
    return np.column_stack([x, y, angles])

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620  # 10s safety margin before 630s limit
    
    # Build grids
    sub_x, sub_y = _build_subgrid(step=3)
    
    # Use the Pogossnitsky-Scheybman analytic path as initial guess
    n_ctrl = 8
    ps_params = _ps_path(n_ctrl)
    
    # Also try the hand-tuned path
    t = np.linspace(0, 1, n_ctrl)
    x_hand = -2.0 * t
    y_hand = 0.3 * np.sin(np.pi * t)
    angles_hand = 90.0 * t
    hand_params = np.column_stack([x_hand, y_hand, angles_hand])
    
    # Evaluate both on subgrid
    ps_area = _evaluate_path_fast(ps_params, sub_x, sub_y, np.ones(len(sub_x), dtype=bool))
    hand_area = _evaluate_path_fast(hand_params, sub_x, sub_y, np.ones(len(sub_x), dtype=bool))
    
    # Start with the better one
    if ps_area >= hand_area:
        best_params = ps_params.copy()
        best_area_sub = ps_area
    else:
        best_params = hand_params.copy()
        best_area_sub = hand_area
    
    # Hill climb on subgrid
    best_params, best_area_sub = _hill_climb(best_params, sub_x, sub_y, deadline, n_iter=500)
    
    if time.time() > deadline:
        return best_params
    
    # Try different initial paths
    best_overall = best_params.copy()
    best_overall_area = best_area_sub
    
    # Alternative initial paths
    alt_paths = []
    
    # Path 1: More aggressive y movement
    t1 = np.linspace(0, 1, n_ctrl)
    x1 = -2.0 * t1
    y1 = 0.5 * np.sin(np.pi * t1)
    angles1 = 90.0 * t1
    alt_paths.append(np.column_stack([x1, y1, angles1]))
    
    # Path 2: Slower rotation at start
    t2 = np.linspace(0, 1, n_ctrl)
    x2 = -2.0 * t2
    y2 = 0.4 * np.sin(np.pi * t2)
    angles2 = 90.0 * (t2 ** 0.7)  # slower start
    alt_paths.append(np.column_stack([x2, y2, angles2]))
    
    # Path 3: Faster rotation at start
    t3 = np.linspace(0, 1, n_ctrl)
    x3 = -2.0 * t3
    y3 = 0.4 * np.sin(np.pi * t3)
    angles3 = 90.0 * (t3 ** 1.5)  # faster start
    alt_paths.append(np.column_stack([x3, y3, angles3]))
    
    # Path 4: PS path with different n_ctrl
    ps_params_12 = _ps_path(12)
    alt_paths.append(ps_params_12)
    
    # Path 5: PS path with fewer points
    ps_params_5 = _ps_path(5)
    alt_paths.append(ps_params_5)
    
    for alt in alt_paths:
        if time.time() > deadline:
            break
        area_alt = _evaluate_path_fast(alt, sub_x, sub_y, np.ones(len(sub_x), dtype=bool))
        if area_alt > best_overall_area:
            best_overall = alt.copy()
            best_overall_area = area_alt
    
    # Final hill climb on best path
    best_params, best_area_sub = _hill_climb(best_overall, sub_x, sub_y, deadline, n_iter=300)
    
    if time.time() > deadline:
        return best_params
    
    # Try gradient descent with smooth surrogate if time permits
    if time.time() < deadline - 60:
        gd_params, gd_area = _gradient_descent_smooth(best_params, sub_x, sub_y, deadline, n_iter=100, lr=0.05, beta=20.0)
        if gd_area > best_area_sub:
            best_params = gd_params
            best_area_sub = gd_area
    
    if time.time() > deadline:
        return best_params
    
    # Validate on full grid and do final refinement
    full_x, full_y = _build_grid()
    full_area = _evaluate_path_fast(best_params, full_x, full_y, np.ones(len(full_x), dtype=bool))
    
    # If there's time, do a final refinement on full grid
    if time.time() < deadline - 30:
        best_params, full_area = _hill_climb(best_params, full_x, full_y, deadline, n_iter=100)
    
    return best_params

# EVOLVE-BLOCK-END
