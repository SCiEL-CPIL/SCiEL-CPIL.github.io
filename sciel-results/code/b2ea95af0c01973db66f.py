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

def _points_from_sorted_angles(angles):
    """
    Generate 13 points on the unit circle given sorted angles.
    The input angles are assumed to be sorted in [0, 2*pi).
    """
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def _log_barrier_objective(angles, target_ratio, mu):
    """
    Objective function for log-barrier constrained optimization.
    
    We want to maximize min_area / hull_area.
    Let R = min_area / hull_area.
    We want R >= target_ratio.
    
    The objective to minimize is:
    J(angles) = -target_ratio * log(min_area) + log(hull_area) + mu * Barrier(angles)
    
    Actually, a more stable formulation for "maximize min ratio" is to maximize the minimum area 
    subject to hull_area <= 1 (normalized) or similar. 
    However, the ratio is invariant to scaling. We can fix the scale by fixing the hull area? 
    No, the points are on the unit circle, so hull area is fixed by the angles.
    
    Let's use the following formulation:
    Maximize min_area subject to hull_area = 1? No, hull_area varies with angles.
    
    Alternative: Maximize log(min_area) - log(hull_area).
    This is equivalent to maximizing min_area / hull_area.
    To handle the "min" operation smoothly and enforce the target, we use a barrier.
    
    Let A_min = min(areas).
    Objective = -log(A_min) + log(HullArea) + mu * Sum(log(gaps))
    
    Wait, we want to MAXIMIZE A_min / HullArea.
    So we MINIMIZE -log(A_min / HullArea) = log(HullArea) - log(A_min).
    
    The "min" operation is non-differentiable. We approximate it with a soft-min or use the fact that 
    at the optimum, the gradient of the min is non-zero only for the active constraints.
    However, standard gradient descent on the hard min is tricky.
    
    Let's use the "target" approach:
    We maintain a target ratio T.
    We want to find angles such that all triangle areas / hull_area >= T.
    If we can satisfy this, we increase T.
    
    Objective for fixed T:
    Minimize sum_{i} max(0, T * HullArea - Area_i)^2 + mu * Barrier(angles)
    
    This is a least-squares barrier problem. It is differentiable almost everywhere.
    
    Barrier(angles) ensures angles are sorted and distinct.
    Let gaps = diff(angles) and wrap gap.
    Barrier = -sum(log(gaps))
    
    So we minimize:
    F(angles) = sum_{i} max(0, T * H - A_i)^2 - mu * sum(log(gaps_i))
    
    If F can be made small (close to 0), it means all areas >= T*H.
    Then we increase T.
    """
    
    # 1. Compute points and areas
    pts = _points_from_sorted_angles(angles)
    H = _hull_area(pts)
    if H < 1e-9:
        return 1e10
    
    A = _triangle_areas(pts)
    
    # 2. Compute Barrier term for sorted angles
    # Angles are in [0, 2*pi). We need to ensure they are strictly increasing.
    # The optimizer might violate this if not constrained. 
    # We use a barrier on the gaps.
    # gaps = np.diff(angles)
    # wrap_gap = angles[0] + 2*pi - angles[-1]
    # We need all gaps > 0.
    
    # To handle the periodicity and sorting, we assume the input angles are sorted.
    # If the optimizer tries to cross them, the barrier should explode.
    # Let's compute gaps carefully.
    # If angles are not sorted, the "gaps" concept breaks.
    # We should enforce sorting via the barrier.
    # A robust way: sort the angles first? No, that makes the objective non-smooth.
    # Instead, we define the barrier based on the *sorted* order, but if the input is unsorted, 
    # we penalize it heavily or just assume the optimizer stays in the sorted manifold.
    # Given the symmetry, we can fix angle[0] = 0 and optimize angle[1..12] in (0, 2*pi).
    # This reduces dimensionality and ensures ordering if we constrain angle[i] < angle[i+1].
    
    # Let's re-parameterize:
    # theta[0] = 0
    # theta[1] ... theta[12] in (0, 2*pi)
    # Constraints: 0 < theta[1] < ... < theta[12] < 2*pi
    
    # The input `angles` to this function should be the full vector or the reduced vector?
    # Let's use the full vector for simplicity in the wrapper, but internally handle the barrier.
    
    # If we just use the full vector, we need to ensure the barrier pushes them into sorted order.
    # A common trick: use the sorted angles for geometry, but penalize the *unsorted* nature?
    # No, that's messy.
    
    # Better approach: 
    # The function `_log_barrier_objective` takes `angles` which are assumed to be the 
    # *unconstrained* variables. But we want to optimize over the sorted manifold.
    # Let's change the strategy: 
    # We optimize over `x` of size N-1.
    # x[0] = gap_1, ..., x[N-2] = gap_{N-1}, x[N-1] = gap_wrap?
    # Sum of gaps = 2*pi.
    # This is a simplex constraint.
    
    # Let's stick to the simpler formulation for now:
    # Optimize `angles` directly, but add a huge penalty if they are not sorted.
    # Or, better, use the sorted angles for calculation and a barrier on the gaps.
    
    # Let's assume the caller passes angles that are *nearly* sorted.
    # We will sort them for the geometric calculation to ensure valid hull/areas.
    # But we need the gradient to push them towards sorted order.
    # If we sort inside the function, the gradient is zero for permutations (non-differentiable).
    
    # Correct Approach for this specific step:
    # We will use a reduced parameterization in `construct`.
    # Here, we just compute the objective for a given set of *sorted* angles.
    # The barrier term is -sum(log(gaps)).
    
    sorted_angles = np.sort(angles)
    
    # Calculate gaps
    gaps = np.diff(sorted_angles)
    wrap_gap = sorted_angles[0] + 2 * np.pi - sorted_angles[-1]
    all_gaps = np.append(gaps, wrap_gap)
    
    # Barrier: -sum(log(gaps))
    # If any gap <= 0, this is -inf. We add a small epsilon.
    log_gaps = np.log(np.maximum(all_gaps, 1e-12))
    barrier_term = -np.sum(log_gaps)
    
    # Penalty term for violating the target ratio
    # We want A_i >= T * H
    # Violation_i = max(0, T * H - A_i)
    # Sum of squares
    violations = np.maximum(0, target_ratio * H - A)
    penalty = np.sum(violations ** 2)
    
    return penalty + mu * barrier_term

def _optimize_with_barrier(start_angles, target_ratio, mu, time_limit):
    """
    Run L-BFGS-B to minimize the barrier objective for a fixed target ratio.
    """
    start_time = time.time()
    
    def obj(x):
        # Check time
        if time.time() > time_limit:
            return 1e10
        return _log_barrier_objective(x, target_ratio, mu)
    
    # Bounds for angles: [0, 2*pi]
    bounds = [(0, 2 * np.pi) for _ in range(N)]
    
    try:
        res = minimize(
            obj,
            start_angles,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x
    except Exception:
        return start_angles

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial guess: Regular polygon
    init_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    best_score = calculate_score(_points_from_sorted_angles(init_angles))
    best_angles = init_angles.copy()
    
    # Strategy:
    # 1. Start with a conservative target ratio T.
    # 2. Use barrier optimization to find angles that satisfy A_i >= T * H.
    # 3. If successful (penalty ~ 0), increase T.
    # 4. If failed, decrease mu or adjust T.
    
    # Estimate initial score to set T
    # Regular 13-gon score is approx 0.02-0.03? 
    # Let's calculate it.
    reg_pts = _points_from_sorted_angles(init_angles)
    reg_score = calculate_score(reg_pts)
    
    # We want to maximize the score. 
    # The barrier method finds a feasible point for T.
    # We can binary search for the maximum T.
    
    # However, the feasible set might be empty for high T.
    # Let's try to push T up iteratively.
    
    current_target = reg_score * 0.9  # Start slightly below current
    mu = 1.0 # Barrier parameter
    
    # We will iterate:
    # Try to find angles for current_target.
    # If min_area/hull_area > current_target, we can increase current_target.
    
    num_iterations = 20
    time_per_iter = (deadline - start_time) / (num_iterations + 5)
    
    for i in range(num_iterations):
        if time.time() > deadline - 10:
            break
        
        # Optimize for current_target
        # Use current best angles as start
        new_angles = _optimize_with_barrier(best_angles, current_target, mu, deadline - 10)
        
        # Sort and calculate new score
        new_angles_sorted = np.sort(new_angles)
        new_pts = _points_from_sorted_angles(new_angles_sorted)
        new_score = calculate_score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_angles = new_angles_sorted
        
        # Check if we achieved the target
        # If new_score >= current_target * (1 + epsilon), we can increase target
        # Actually, if the optimizer succeeded, the penalty should be low, meaning new_score >= current_target.
        # Let's check the actual min area ratio.
        
        if new_score >= current_target * 0.99:
            # We are close to or above target. Increase target.
            # Geometric increase
            current_target *= 1.05
            # Decrease mu to allow more freedom? Or keep it constant?
            # Keeping mu constant maintains the convexity constraint strength.
        else:
            # We didn't reach the target. 
            # Maybe target is too high, or mu is too strong?
            # Try decreasing target slightly to find a feasible point, then increase.
            # Or just keep the current best and try a different start?
            # For now, let's just keep the current target and try again with a different random perturbation
            # to escape local minima.
            
            # Perturb best angles
            perturb = 0.01 * np.random.randn(N)
            perturbed_angles = best_angles + perturb
            # Re-sort to maintain manifold
            perturbed_angles = np.sort(perturbed_angles)
            
            new_angles = _optimize_with_barrier(perturbed_angles, current_target, mu, deadline - 10)
            new_angles_sorted = np.sort(new_angles)
            new_pts = _points_from_sorted_angles(new_angles_sorted)
            new_score = calculate_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_angles = new_angles_sorted
            
            if new_score >= current_target * 0.99:
                current_target *= 1.02
            else:
                # Stuck. Try reducing mu to see if we can move more freely?
                # If mu is too high, the points are pushed to be equidistant (regular polygon).
                # If mu is low, they might cluster.
                # Let's try lowering mu slightly.
                mu *= 0.9

    # Final Polish
    # Use the best angles found and do a final standard minimization on -score
    # to squeeze out any remaining gain.
    
    if time.time() < deadline - 5:
        def final_obj(x):
            pts = _points_from_sorted_angles(np.sort(x))
            return -calculate_score(pts)
        
        try:
            res_final = minimize(
                final_obj,
                best_angles,
                method='L-BFGS-B',
                bounds=[(0, 2 * np.pi) for _ in range(N)],
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            final_angles = np.sort(res_final.x)
            final_pts = _points_from_sorted_angles(final_angles)
            final_score = calculate_score(final_pts)
            
            if final_score > best_score:
                best_score = final_score
                best_angles = final_angles
        except Exception:
            pass

    # Convert best angles to points
    best_points = _points_from_sorted_angles(np.sort(best_angles))
    
    # One final score calculation
    final_score = calculate_score(best_points)
    
    # Fallback
    if final_score <= 0 or not np.isfinite(final_score):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        final_score = calculate_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(calculate_score(pts))
