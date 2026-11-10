import numpy as np
from scipy.spatial.distance import pdist
import time
import scipy.optimize as so

# EVOLVE-BLOCK-START
def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _sqp_maximize_min_dist(points, max_dist_sq, max_iter=100, tol=1e-8):
    """
    Maximize the minimum distance between points, subject to the constraint 
    that all pairwise distances are <= max_dist_sq.
    Uses Sequential Quadratic Programming (SLSQP).
    """
    n = len(points)
    x0 = points.flatten()
    
    # Objective: maximize min distance.
    # Equivalent to minimizing -min distance.
    # To make it smooth, we can use a log-sum-exp approximation or just use the min.
    # SLSQP can handle non-smooth objectives to some extent, but a smooth proxy is better.
    # Let's try maximizing the sum of logs of distances (geometric mean) as a proxy,
    # or simply minimizing the negative of the minimum distance.
    # Actually, a common trick is to introduce a variable r and maximize r
    # subject to dist(i,j) >= r for all i,j.
    # This is a non-convex problem generally, but SLSQP is a local solver.
    
    # Variables: x (2n coords) + r (1 scalar)
    # Total 2n + 1 variables.
    
    def objective(v):
        # v[0:2n] are points, v[2n] is r
        # We want to maximize r, so minimize -r
        return -v[2*n]
        
    def jac(v):
        # Gradient of -r w.r.t v is 0 for points, -1 for r
        grad = np.zeros(2*n + 1)
        grad[2*n] = -1.0
        return grad

    constraints = []
    
    # 1. Max distance constraint: ||p_i - p_j||^2 <= max_dist_sq
    # This is a convex constraint.
    # We need to define it for all pairs.
    # To reduce the number of constraints, we can define them dynamically or use a vectorized approach.
    # SLSQP allows a list of constraint dicts.
    # However, creating 120 constraints might be slow if re-evaluated frequently.
    # We can define a single constraint function that returns the vector of violations?
    # SLSQP expects each constraint dict to have 'fun' returning a scalar or vector?
    # Actually, 'fun' in SLSQP constraint dict can return an array for inequality constraints?
    # No, standard SLSQP expects scalar for each constraint in the list, OR a single constraint with 'fun' returning array?
    # Documentation: "The function fun returns a scalar or an array of values."
    # If it returns an array, it's treated as multiple constraints?
    # Let's check: scipy.optimize.minimize with method='SLSQP'
    # constraints: sequence of Constraint objects or dicts.
    # If 'fun' returns an array, it is treated as a vector of constraints?
    # Yes, "If fun returns an array, it is treated as a vector of constraints."
    
    def max_dist_constraint(v):
        pts = v[:2*n].reshape(n, 2)
        # Compute all pairwise squared distances
        # We want dist_sq <= max_dist_sq => dist_sq - max_dist_sq <= 0
        # SLSQP inequality constraint is fun(x) >= 0.
        # So we want max_dist_sq - dist_sq >= 0.
        d = pdist(pts)
        return max_dist_sq - d**2

    cons_max_dist = {'type': 'ineq', 'fun': max_dist_constraint}
    constraints.append(cons_max_dist)
    
    # 2. Min distance constraint: ||p_i - p_j||^2 >= r^2
    # dist_sq - r^2 >= 0
    def min_dist_constraint(v):
        pts = v[:2*n].reshape(n, 2)
        r = v[2*n]
        d = pdist(pts)
        return d**2 - r**2

    cons_min_dist = {'type': 'ineq', 'fun': min_dist_constraint}
    constraints.append(cons_min_dist)
    
    # Bounds: 
    # Points can be anywhere, but we should bound them to prevent divergence.
    # The cluster should be within a box of size related to sqrt(max_dist_sq).
    # Let's say [-10, 10] is a safe bound for unit-scaled problems.
    # r should be positive.
    bounds = [(-10, 10)] * (2*n) + [(1e-6, 10)]
    
    # Initial guess for r: small positive value
    v0 = np.append(x0, 1e-4)
    
    try:
        res = so.minimize(objective, v0, method='SLSQP', jac=jac, bounds=bounds, 
                          constraints=constraints, options={'maxiter': max_iter, 'ftol': tol})
        if res.success:
            return res.x
        else:
            # If SLSQP fails, return best found or original
            # Check if the result is feasible
            # For simplicity, return original if it fails, or try to extract x
            # Let's return the original points if it fails significantly
            return x0
    except Exception:
        return x0

def _bisection_solve(initial_points, time_limit=500.0):
    """
    Use bisection on the ratio R = r_max / d_max.
    Actually, we can fix the max distance to 1 (by scaling) and maximize the min distance.
    If we find a configuration with min_dist = r* and max_dist = 1, the ratio is r*.
    Then we scale the points to have max_dist = 1.
    
    Algorithm:
    1. Start with a candidate set of points.
    2. Normalize to max_dist = 1.
    3. Calculate current min_dist.
    4. Use SLSQP to maximize min_dist subject to max_dist <= 1.
    5. If min_dist increases, update.
    6. Repeat or use multiple starts.
    """
    n = len(initial_points)
    best_points = initial_points.copy()
    best_ratio = 0.0
    
    start_time = time.time()
    
    # Try to improve the initial configuration
    for _ in range(10): # Max 10 SLSQP iterations
        if time.time() - start_time > time_limit:
            break
        
        # Normalize current best points to have max_dist = 1
        d = pdist(best_points)
        d_max = d.max()
        if d_max > 1e-9:
            normalized_points = best_points / d_max
        else:
            normalized_points = best_points
        
        # Run SLSQP to maximize min distance with max_dist <= 1
        # max_dist_sq = 1.0
        result_x = _sqp_maximize_min_dist(normalized_points, 1.0, max_iter=50, tol=1e-6)
        
        if result_x is not None and len(result_x) == 2*n + 1:
            new_points = result_x[:2*n].reshape(n, 2)
            
            # Verify feasibility and calculate score
            d_new = pdist(new_points)
            d_min_new = d_new.min()
            d_max_new = d_new.max()
            
            # Check if max_dist is actually <= 1 (within tolerance)
            if d_max_new <= 1.0 + 1e-5 and d_min_new > 1e-9:
                ratio_new = d_min_new / d_max_new
                if ratio_new > best_ratio:
                    best_ratio = ratio_new
                    best_points = new_points.copy()
            else:
                # If infeasible, maybe SLSQP drifted. Try to project or ignore.
                pass

    return best_points, best_ratio

def _generate_initial_points(seed=0):
    """Generate multiple structured initial configurations."""
    n = 16
    rng = np.random.RandomState(seed)
    candidates = []

    # 1. Regular polygon + center (8-gon + inner 8-gon scaled)
    angles_outer = 2 * np.pi * np.arange(8) / 8
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    inner = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    candidates.append(np.vstack([outer, inner]))

    # 2. 4x4 grid
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid = np.column_stack(np.meshgrid(xs, ys, indexing='ij')).reshape(-1, 2)
    candidates.append(grid)

    # 3. Two concentric 8-gons
    angles1 = 2 * np.pi * np.arange(8) / 8
    angles2 = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    r1 = 1.0
    r2 = 0.6
    p1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    p2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    candidates.append(np.vstack([p1, p2]))

    # 4. Hexagonal-ish arrangement: 1 center + 6 ring + 9 outer
    center = np.array([[0.0, 0.0]])
    ring1_angles = 2 * np.pi * np.arange(6) / 6
    ring1 = 0.8 * np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    ring2_angles = 2 * np.pi * np.arange(9) / 9
    ring2 = 1.5 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    candidates.append(np.vstack([center, ring1, ring2]))

    # 5. Random
    candidates.append(rng.randn(n, 2) * 0.5)

    # 6. Another structured: 2x8 rectangle with jitter
    xs = np.tile(np.array([-1, 1]), 8)
    ys = np.linspace(-1, 1, 16)
    rect = np.column_stack([xs, ys])
    candidates.append(rect + rng.randn(16, 2) * 0.05)
    
    # 7. Random with larger spread
    candidates.append(rng.randn(n, 2) * 2.0)

    return candidates

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    deadline = time.time() + 580.0  # Safety margin before 630s kill
    
    # Generate initial candidates
    initial_points_list = _generate_initial_points(seed=42)
    
    best_points = initial_points_list[0].copy()
    best_score = _compute_score(best_points)
    
    # Run bisection/SQP from each initial configuration
    for i, init_pts in enumerate(initial_points_list):
        if time.time() > deadline - 30.0:
            break
        
        time_budget = min(50.0, deadline - time.time() - 10.0)
        if time_budget < 5.0:
            break
        
        try:
            result_pts, result_ratio = _bisection_solve(init_pts, time_limit=time_budget)
            result_score = result_ratio ** 2
            if result_score > best_score:
                best_points = result_pts
                best_score = result_score
        except Exception as e:
            # If SLSQP crashes, skip this candidate
            pass

    # Final refinement: run a short SLSQP from the best point found
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        try:
            # Normalize best points
            d = pdist(best_points)
            d_max = d.max()
            if d_max > 1e-9:
                normalized_best = best_points / d_max
            else:
                normalized_best = best_points
            
            result_x = _sqp_maximize_min_dist(normalized_best, 1.0, max_iter=100, tol=1e-7)
            if result_x is not None and len(result_x) == 2*n + 1:
                new_points = result_x[:2*n].reshape(n, 2)
                d_new = pdist(new_points)
                d_min_new = d_new.min()
                d_max_new = d_new.max()
                if d_max_new <= 1.0 + 1e-5 and d_min_new > 1e-9:
                    new_score = (d_min_new / d_max_new) ** 2
                    if new_score > best_score:
                        best_points = new_points
                        best_score = new_score
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
