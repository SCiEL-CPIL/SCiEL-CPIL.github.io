import numpy as np
import time
import scipy.spatial as spat
from scipy.optimize import minimize
from itertools import combinations

N = 13
# Pre-compute triangle indices
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])

def _triangle_areas(points):
    """Compute areas of all C(N,3) triangles."""
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-9

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    """
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _barrier_objective(angles, target_ratio, mu):
    """
    Objective function for log-barrier optimization.
    Maximizes the minimum area ratio subject to the constraint that all areas are >= target_ratio * hull_area.
    
    Parameters:
    angles: (N,) array of sorted angles on [0, 2*pi)
    target_ratio: The target minimum area / hull area ratio
    mu: Barrier parameter (inverse of penalty strength)
    
    Returns:
    Objective value to minimize.
    """
    # Ensure angles are sorted and within [0, 2*pi)
    # The optimizer might drift, so we normalize
    a = np.sort(angles)
    # Shift to center around 0 for numerical stability? No, just normalize to [0, 2*pi)
    # Actually, if we just use cos/sin, the absolute offset doesn't matter.
    # But the barrier calculation needs the actual points.
    
    # To ensure convexity and avoid point swapping issues in the barrier,
    # we rely on the fact that for points on the unit circle in sorted angular order,
    # the polygon is convex.
    
    pts = np.column_stack((np.cos(a), np.sin(a)))
    
    # Calculate areas
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    # Hull area for regular-ish polygon on unit circle
    # For points on unit circle, hull area is sum of 0.5 * sin(angle_diff) for consecutive points
    # Assuming sorted angles, consecutive differences are a[i+1]-a[i] and a[0]+2pi-a[-1]
    diffs = np.diff(a)
    diffs = np.append(diffs, a[0] + 2*np.pi - a[-1])
    # Ensure diffs are positive (sortedness)
    diffs = np.abs(diffs)
    # If points are very close, sin might be small, but area is sum of 0.5*sin(diff)
    hull_area = 0.5 * np.sum(np.sin(diffs))
    
    if hull_area < 1e-9:
        return 1e6
    
    # Constraint: areas_i >= target_ratio * hull_area
    # Violation: max(0, target_ratio * hull_area - areas_i)
    # Log barrier: -mu * sum(log(areas_i - target_ratio * hull_area))
    # We need to ensure arguments are positive.
    
    constraints_val = areas - target_ratio * hull_area
    
    # If any constraint is violated (constraints_val < 0), we return a large penalty
    # to guide the optimizer back into the feasible region.
    if np.any(constraints_val <= 0):
        # Penalty proportional to violation
        min_val = np.min(constraints_val)
        return 1e6 - min_val * 1e4 # Large positive number to minimize (so we want to maximize min_val)
        # Wait, minimize function. We want to push towards feasible.
        # If infeasible, return large positive.
        # Let's just return a large penalty based on how far infeasible.
        return 1e6 - 1000 * min_val
    
    # Feasible region
    # Objective: Minimize - (log_barrier) ? 
    # Actually, we want to maximize the minimum ratio.
    # The standard log barrier method for maximizing f(x) subject to g(x) >= 0 is:
    # Minimize -f(x) - mu * sum(log(g_i(x)))
    # Here f(x) is not explicitly in the objective if we are just checking feasibility for a fixed target.
    # But we want to *find* the maximum target.
    
    # Strategy: 
    # 1. Fix target_ratio.
    # 2. Check if there exists an angle configuration satisfying areas >= target_ratio * hull_area.
    # 3. If yes, increase target_ratio.
    # 4. If no, decrease target_ratio.
    
    # However, we can also just optimize for the "slack" in the constraints.
    # Let's define the objective as minimizing the negative of the smallest slack, penalized by the barrier.
    # Actually, a simpler approach for this specific "maximize min" problem:
    # Minimize: -min(areas) / hull_area + (1/mu) * sum(log(hull_area * target_ratio - areas_i)) ? No.
    
    # Let's stick to the standard barrier for feasibility of a given target, and use binary search on target.
    # But that requires many calls.
    
    # Alternative: Minimize the negative of the minimum area ratio directly, but use the barrier to keep points "well-spread"?
    # No, the directive says: "barrier term penalizes any triangle area falling below a target threshold".
    
    # Let's implement the objective as:
    # J(angles) = - (min_area / hull_area) + (1/mu) * sum( log( max(0, target_ratio * hull_area - areas_i) + epsilon ) )
    # This is not quite right.
    
    # Let's use the standard formulation:
    # We want to maximize phi(angles) = min(areas_i) / hull_area.
    # We can approximate this by minimizing:
    # - log( sum( exp( -areas_i / T ) ) ) / hull_area ... this is the soft min again.
    
    # The specific strategy requested: "log-barrier constrained optimization ... where the barrier term penalizes any triangle area falling below a target threshold".
    # This implies we are solving:
    # Maximize: 0 (or some constant)
    # Subject to: areas_i >= target_ratio * hull_area for all i.
    # Using barrier method, we minimize:
    # B(angles) = -mu * sum( log( areas_i - target_ratio * hull_area ) )
    # If we can find a solution, the target is feasible.
    
    # So the objective to minimize is:
    # -mu * sum( log( areas_i - target_ratio * hull_area ) )
    
    # If we minimize this, we are trying to maximize the product of the slacks (areas_i - target*hull).
    # This drives all slacks to be as large as possible? No, it drives them to be balanced and positive.
    # If the minimum slack is 0, the log goes to -infinity, so the objective goes to +infinity.
    # So minimizing this objective pushes the configuration into the interior of the feasible region.
    
    # If we can minimize this successfully (objective doesn't blow up), the target is feasible.
    
    return -mu * np.sum(np.log(constraints_val))

def _optimize_angles(angles_init, target_ratio, time_budget):
    """
    Run optimization to find angles satisfying the barrier constraint for a given target ratio.
    Returns (success, best_angles, best_obj)
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initial guess: regular polygon
    a0 = np.sort(angles_init)
    
    best_obj = np.inf
    best_a = a0.copy()
    success = False
    
    # Annealing mu? Or just fixed mu?
    # Smaller mu -> closer to boundary. Larger mu -> more central.
    # We want to check feasibility, so we want to see if we can get into the feasible region.
    # Let's try a few mu values.
    
    mus = [10.0, 5.0, 1.0, 0.5, 0.1]
    
    for mu in mus:
        if time.time() > deadline:
            break
        
        def obj(a):
            return _barrier_objective(a, target_ratio, mu)
        
        try:
            res = minimize(obj, a0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            obj_val = res.fun
            if obj_val < best_obj:
                best_obj = obj_val
                best_a = res.x.copy()
                
            # Check if we are feasible
            # Feasible if all constraints are satisfied
            pts = np.column_stack((np.cos(np.sort(best_a)), np.sin(np.sort(best_a))))
            v1 = pts[_j_arr] - pts[_i_arr]
            v2 = pts[_k_arr] - pts[_i_arr]
            det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(det)
            
            diffs = np.diff(np.sort(best_a))
            diffs = np.append(diffs, np.sort(best_a)[0] + 2*np.pi - np.sort(best_a)[-1])
            hull_area = 0.5 * np.sum(np.sin(np.abs(diffs)))
            
            if hull_area > 1e-9:
                min_ratio = np.min(areas) / hull_area
                if min_ratio >= target_ratio:
                    success = True
                    # Update best if this is better (higher min ratio)
                    # We are just checking feasibility for now
                    break # If feasible, we can stop this mu and maybe try higher target?
                    
            # Update start point for next mu
            a0 = res.x
        except Exception:
            pass
            
    return success, best_a, best_obj

def _find_max_target(angles_init, time_budget):
    """
    Binary search for the maximum target ratio.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initial bounds
    lo = 0.0
    hi = 0.5 # Max possible ratio is much lower, but start wide
    
    # Regular 13-gon score is approx?
    # Area of regular n-gon: 0.5 * n * sin(2*pi/n)
    # Min triangle area: formed by 3 consecutive points?
    # Area = 0.5 * sin(2*pi/n) * sin(2*pi/n) * sin(2*pi - 4*pi/n) ? No.
    # Points on unit circle. Triangle area = 0.5 * |sin(a-b) + sin(b-c) + sin(c-a)|
    # For consecutive points: 0.5 * |sin(2pi/n) + sin(2pi/n) + sin(2pi - 4pi/n)|
    # = 0.5 * |2 sin(2pi/n) - sin(4pi/n)|
    # For n=13, 2pi/13 ~ 0.48
    # sin(0.48) ~ 0.46
    # 2*0.46 - sin(0.96) ~ 0.92 - 0.82 = 0.1
    # Area ~ 0.05
    # Hull area ~ 0.5 * 13 * sin(0.48) ~ 6.5 * 0.46 ~ 3.0
    # Ratio ~ 0.05/3.0 ~ 0.016
    
    # Let's calculate exact score for regular polygon to set hi
    angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts_reg = np.column_stack((np.cos(angles_reg), np.sin(angles_reg)))
    score_reg = calculate_score(pts_reg)
    
    hi = score_reg * 1.5
    lo = score_reg * 0.5
    
    best_angles = angles_reg
    best_score = score_reg
    
    # Binary search iterations
    max_iters = 20
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        mid = (lo + hi) / 2
        
        # Try to find a configuration satisfying ratio >= mid
        # Allocate time for this check
        time_per_check = (deadline - time.time()) / (max_iters - i)
        
        success, angles_found, _ = _optimize_angles(angles_reg, mid, time_per_check)
        
        if success:
            lo = mid
            # Update best if this configuration is good
            pts_found = np.column_stack((np.cos(angles_found), np.sin(angles_found)))
            score_found = calculate_score(pts_found)
            if score_found > best_score:
                best_score = score_found
                best_angles = angles_found
        else:
            hi = mid
            
    return best_angles, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial guess: Regular 13-gon
    angles_init = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Run the search
    best_angles, best_score = _find_max_target(angles_init, deadline - start_time - 5.0)
    
    # Convert angles to points
    best_points = np.column_stack((np.cos(best_angles), np.sin(best_angles)))
    
    # Final polish: Try to improve the best point found using the original smooth min objective
    # This helps if the barrier method got stuck near the boundary
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            def obj_smooth(x):
                pts = x.reshape(N, 2)
                v1 = pts[_j_arr] - pts[_i_arr]
                v2 = pts[_k_arr] - pts[_i_arr]
                det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                areas = 0.5 * np.abs(det)
                
                hull_area = _hull_area(pts)
                if hull_area < 1e-6:
                    return 1e6
                
                min_a = np.min(areas)
                # Soft min
                T = 0.01
                exp_terms = np.exp(-(areas - min_a) / T)
                sum_exp = np.sum(exp_terms)
                smooth_min_val = min_a - T * np.log(sum_exp)
                
                return -smooth_min_val / hull_area
            
            res = minimize(obj_smooth, best_points.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            pts_final = res.x.reshape(N, 2)
            score_final = calculate_score(pts_final)
            
            if score_final > best_score:
                best_score = score_final
                best_points = pts_final
        except Exception:
            pass

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(calculate_score(pts))
