import numpy as np
import time
import scipy.optimize as opt
from itertools import combinations

N = 13

# Pre-compute triangle indices
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])
_n_triples = len(_triple_idx)

def _get_points_from_angles(angles):
    """Convert angles to points on the unit circle."""
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _triangle_areas_from_points(points):
    """Compute areas of all C(N,3) triangles."""
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)

def _hull_area(points):
    """Compute convex hull area."""
    # Since points are on the unit circle, if sorted by angle, 
    # the hull is the polygon formed by them.
    # Area of a polygon with vertices (x_i, y_i) is 0.5 * |sum(x_i*y_{i+1} - x_{i+1}*y_i)|
    # But we need to be careful with the order. 
    # The input points are already on the circle, but not necessarily sorted.
    # However, for the score calculation, we just need the area of the convex hull.
    # For points on a circle, the convex hull is the polygon connecting them in angular order.
    # If we assume the points are sorted by angle in the optimization, we can compute area directly.
    # But for robustness, let's use the shoelace formula on the sorted points.
    # Actually, the standard ConvexHull is robust.
    # Given the constraint is "on unit circle", the hull area is determined by the angular gaps.
    # Area = 0.5 * sum(sin(angle_diff)) for consecutive points in angular order?
    # Yes, area of triangle (0,0, P_i, P_{i+1}) is 0.5 * sin(theta_{i+1} - theta_i).
    # The total area is the sum of these triangles.
    pass

def _score_angles(angles):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    angles: array of N angles.
    """
    points = _get_points_from_angles(angles)
    
    # Calculate Hull Area
    # Sort angles to compute hull area correctly
    sorted_angles = np.sort(angles)
    # Add 2pi to the first angle to close the loop
    sorted_angles_ext = np.append(sorted_angles, sorted_angles[0] + 2 * np.pi)
    diffs = np.diff(sorted_angles_ext)
    # Area of each triangle (0, P_i, P_{i+1}) is 0.5 * sin(diff)
    # Summing these gives the total area of the convex hull (polygon inscribed in unit circle)
    hull_area = 0.5 * np.sum(np.sin(diffs))
    
    if hull_area <= 1e-9:
        return 0.0

    # Calculate Triangle Areas
    areas = _triangle_areas_from_points(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _negative_soft_min_score(angles, beta=50.0):
    """
    Objective function to minimize: -soft_min_score.
    Uses a soft-min approximation of the minimum triangle area.
    """
    points = _get_points_from_angles(angles)
    
    # Hull area
    sorted_angles = np.sort(angles)
    sorted_angles_ext = np.append(sorted_angles, sorted_angles[0] + 2 * np.pi)
    diffs = np.diff(sorted_angles_ext)
    hull_area = 0.5 * np.sum(np.sin(diffs))
    
    if hull_area <= 1e-9:
        return 1e10

    areas = _triangle_areas_from_points(points)
    
    # Soft min
    max_a = np.max(areas)
    exp_neg = np.exp(-beta * (areas - max_a))
    log_sum = np.log(np.sum(exp_neg)) + beta * max_a
    soft_min_val = log_sum / beta
    
    return -soft_min_val / hull_area

def _gradient_negative_soft_min(angles, beta=50.0):
    """
    Gradient of the negative soft-min score w.r.t. angles.
    """
    points = _get_points_from_angles(angles)
    c = np.cos(angles)
    s = np.sin(angles)
    
    # Hull area and its gradient
    sorted_angles = np.sort(angles)
    sorted_angles_ext = np.append(sorted_angles, sorted_angles[0] + 2 * np.pi)
    diffs = np.diff(sorted_angles_ext)
    hull_area = 0.5 * np.sum(np.sin(diffs))
    
    if hull_area <= 1e-9:
        return np.zeros_like(angles)

    areas = _triangle_areas_from_points(points)
    max_a = np.max(areas)
    exp_neg = np.exp(-beta * (areas - max_a))
    weights = exp_neg / np.sum(exp_neg)
    
    # Gradient of areas w.r.t. angles
    # Area of triangle (i, j, k) = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
    # Let's compute the signed double area D = x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)
    # x = cos, y = sin
    # dD/dtheta_i = -sin_i(y_j - y_k) + cos_i(y_k - y_j) = -sin_i(y_j - y_k) - cos_i(y_j - y_k) ?
    # Let's derive properly.
    # D = x_i y_j - x_i y_k - x_j y_i + x_j y_k + x_k y_i - x_k y_j
    # dD/dtheta_i = -s_i (y_j - y_k) + c_i (y_k - y_j) = -s_i (y_j - y_k) - c_i (y_j - y_k)
    # Wait. d(x_i)/dtheta_i = -s_i. d(y_i)/dtheta_i = c_i.
    # Term with i: x_i y_j -> -s_i y_j. -x_i y_k -> -(-s_i) y_k = s_i y_k. -x_j y_i -> -x_j c_i. x_k y_i -> x_k c_i.
    # So dD/dtheta_i = -s_i y_j + s_i y_k - x_j c_i + x_k c_i = s_i (y_k - y_j) + c_i (x_k - x_j)
    
    # This is getting complicated to vectorize perfectly with signs.
    # Let's use numerical differentiation or a simpler analytic form for the gradient of the area ratio.
    # Actually, for the soft-min, the gradient is weighted sum of gradients of individual areas.
    
    # Let's compute gradient of each triangle area w.r.t. angles.
    # Area = 0.5 * |D|. Assuming D > 0 (consistent orientation), dArea/dtheta = 0.5 * dD/dtheta.
    # We can compute dD/dtheta for all points involved in each triangle.
    
    grad_areas = np.zeros((N, 1)) # Placeholder, will compute full grad vector
    
    # We need grad of soft_min_val / hull_area.
    # f = A_soft / H
    # grad f = (grad A_soft * H - A_soft * grad H) / H^2
    
    # grad A_soft = sum(weights * grad Area_t)
    # grad H = sum over hull edges... this is tricky because H depends on sorted angles.
    # If we assume angles are sorted, H = 0.5 * sum(sin(theta_{i+1} - theta_i)).
    # dH/dtheta_i = 0.5 * (cos(theta_i - theta_{i-1}) - cos(theta_{i+1} - theta_i))
    
    # Let's assume angles are sorted for the gradient calculation of H.
    # If not sorted, the mapping from angles to sorted positions is a permutation.
    # For simplicity in gradient, we can sort angles first, compute gradient in sorted space, then unsort.
    
    sort_idx = np.argsort(angles)
    unsort_idx = np.argsort(sort_idx)
    angles_sorted = angles[sort_idx]
    
    # Compute gradient of H w.r.t. sorted angles
    # H = 0.5 * sum_{i=0}^{N-1} sin(a_{i+1} - a_i) where a_N = a_0 + 2pi
    # dH/da_i = 0.5 * (cos(a_i - a_{i-1}) - cos(a_{i+1} - a_i))
    # Note: indices are modulo N.
    a_prev = np.roll(angles_sorted, 1)
    a_next = np.roll(angles_sorted, -1)
    # Handle wrap-around for the last point
    # The term for a_{N-1} involves a_0 + 2pi.
    # sin(a_0 + 2pi - a_{N-1}) = sin(a_0 - a_{N-1})
    # cos(a_0 + 2pi - a_{N-1}) = cos(a_0 - a_{N-1})
    # So the formula holds if we treat the angle difference correctly.
    
    diff_prev = angles_sorted - a_prev # a_i - a_{i-1}
    diff_next = a_next - angles_sorted # a_{i+1} - a_i
    
    # For the last element, a_next is a_0. We need a_0 + 2pi - a_{N-1}.
    # diff_next for last element is a_0 - a_{N-1}. This is negative if a_{N-1} > a_0.
    # cos is even, sin is odd.
    # The formula dH/da_i = 0.5 * (cos(a_i - a_{i-1}) - cos(a_{i+1} - a_i))
    # For i=N-1: a_{i+1} is a_0. The angle difference should be (a_0 + 2pi) - a_{N-1}.
    # cos((a_0 + 2pi) - a_{N-1}) = cos(a_0 - a_{N-1}).
    # So the standard cosine formula works directly on the raw differences if we are careful?
    # Yes, cos(x + 2pi) = cos(x).
    
    grad_H_sorted = 0.5 * (np.cos(diff_prev) - np.cos(diff_next))
    
    # Map back to original angle order
    grad_H = np.zeros_like(angles)
    grad_H[unsort_idx] = grad_H_sorted
    
    # Now compute grad A_soft
    # A_soft = log(sum(exp(-beta(A_t - A_max)))) / beta
    # dA_soft/dtheta_k = (1/beta) * (sum_t w_t * dA_t/dtheta_k)
    # where w_t = exp(-beta(A_t - A_max)) / sum(...)
    
    # We need dA_t/dtheta_k for all k.
    # A_t = 0.5 * |D_t|.
    # Let's compute D_t and its gradient.
    
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    # Signs
    signs = np.sign(det)
    signs[signs == 0] = 1.0
    
    # Gradient of D_t w.r.t. angles
    # D_t = x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)
    # dD/dtheta_p = -s_p(y_q - y_r) + c_p(y_r - y_q) for p=i, q=j, r=k?
    # Let's re-verify:
    # d/dtheta_i [ x_i(y_j - y_k) ] = -s_i(y_j - y_k)
    # d/dtheta_i [ -x_j y_i ] = -x_j c_i
    # d/dtheta_i [ x_k y_i ] = x_k c_i
    # Sum: -s_i(y_j - y_k) + c_i(x_k - x_j)
    
    # This needs to be accumulated for each point index.
    # Since we have vectorized arrays, we can compute the contribution to each point's gradient.
    
    grad_A_soft = np.zeros_like(angles)
    
    # weights
    max_a = np.max(areas)
    exp_neg = np.exp(-beta * (areas - max_a))
    weights = exp_neg / np.sum(exp_neg)
    
    # For each triangle t, we need to add weight_t * dA_t/dtheta to the gradient.
    # dA_t/dtheta_p = 0.5 * sign_t * dD_t/dtheta_p
    
    # We can do this in a loop over triangles or vectorize the accumulation.
    # N is 13, triples are 286. A loop is fast enough.
    
    for t in range(_n_triples):
        i = _i_arr[t]
        j = _j_arr[t]
        k = _k_arr[t]
        
        w = weights[t] * 0.5 * signs[t]
        
        # dD/dtheta_i
        dDi = -np.sin(angles[i]) * (np.sin(angles[j]) - np.sin(angles[k])) + \
               np.cos(angles[i]) * (np.cos(angles[k]) - np.cos(angles[j]))
        
        # dD/dtheta_j
        dDj = -np.sin(angles[j]) * (np.sin(angles[k]) - np.sin(angles[i])) + \
               np.cos(angles[j]) * (np.cos(angles[i]) - np.cos(angles[k]))
        
        # dD/dtheta_k
        dDk = -np.sin(angles[k]) * (np.sin(angles[i]) - np.sin(angles[j])) + \
               np.cos(angles[k]) * (np.cos(angles[j]) - np.cos(angles[i]))
        
        grad_A_soft[i] += w * dDi
        grad_A_soft[j] += w * dDj
        grad_A_soft[k] += w * dDk
    
    # grad A_soft = (1/beta) * grad_A_soft_raw? No, the formula for soft min derivative includes the 1/beta factor in the definition.
    # A_soft = (1/beta) * log(sum(exp(-beta(A_t - A_max))))
    # dA_soft/dtheta = (1/beta) * (sum(exp(-beta(A_t - A_max)) * (-beta * dA_t/dtheta)) / sum(exp(...)))
    # = - sum(w_t * dA_t/dtheta)
    # Wait, my weights w_t above were exp.../sum.
    # So dA_soft/dtheta = - sum(w_t * dA_t/dtheta).
    
    grad_A_soft = -grad_A_soft # Because of the minus sign in the exponent derivative
    
    # Now combine
    # f = A_soft / H
    # grad f = (grad A_soft * H - A_soft * grad H) / H^2
    
    A_soft_val = np.log(np.sum(exp_neg)) + beta * max_a
    A_soft_val /= beta
    
    grad_f = (grad_A_soft * hull_area - A_soft_val * grad_H) / (hull_area * hull_area)
    
    # We are minimizing -f, so we return -grad_f
    return -grad_f

def _optimize_angles(initial_angles, time_budget, beta=100.0):
    """
    Optimize angles using gradient ascent (via minimize negative objective).
    """
    angles = initial_angles.copy()
    
    # Use L-BFGS-B or similar. We don't have explicit bounds for angles (0 to 2pi),
    # but the function is periodic. We can wrap angles.
    # However, gradient-based methods might drift. 
    # Let's use a simple gradient descent with momentum on the angles directly,
    # wrapping them to [0, 2pi) after each step to maintain consistency.
    
    lr = 0.01
    momentum = np.zeros_like(angles)
    mu = 0.9
    best_score = _score_angles(angles)
    best_angles = angles.copy()
    
    max_iter = 5000
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
            
        obj_val = _negative_soft_min_score(angles, beta)
        grad = _gradient_negative_soft_min(angles, beta)
        
        # Check for NaN
        if not np.all(np.isfinite(grad)):
            break
            
        momentum = mu * momentum + (1 - mu) * grad
        trial = angles + lr * momentum
        
        # Wrap angles to [0, 2pi)
        trial = np.mod(trial, 2 * np.pi)
        
        # Evaluate exact score to decide if we accept the step
        trial_score = _score_angles(trial)
        
        if trial_score > best_score:
            best_score = trial_score
            best_angles = trial.copy()
            angles = trial
            lr *= 0.999
        else:
            lr *= 0.95
            momentum *= 0.5
            if lr < 1e-8:
                break
                
    return best_angles, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -1.0
    best_angles = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Regular polygon
    initial_configs.append(np.linspace(0, 2 * np.pi, N, endpoint=False))
    
    # 2. Random uniform
    np.random.seed(42)
    for _ in range(20):
        initial_configs.append(np.sort(np.random.rand(N) * 2 * np.pi))
        
    # 3. Clustered (to test robustness)
    for _ in range(10):
        base = np.random.rand() * 2 * np.pi
        offsets = np.random.randn(N) * 0.5
        angles = (base + offsets) % (2 * np.pi)
        initial_configs.append(np.sort(angles))
        
    # 4. Two clusters
    for _ in range(10):
        a1 = np.random.rand() * 2 * np.pi
        a2 = a1 + np.pi + np.random.randn() * 0.5
        offs1 = np.random.randn(N//2) * 0.3
        offs2 = np.random.randn(N - N//2) * 0.3
        angles = np.concatenate([a1 + offs1, a2 + offs2])
        angles = np.sort(angles % (2 * np.pi))
        initial_configs.append(angles)
        
    # Optimize each config
    time_per_config = (deadline - start_time - 5.0) / len(initial_configs)
    if time_per_config < 0.5:
        time_per_config = 0.5
        
    for i, init_ang in enumerate(initial_configs):
        if time.time() > deadline - 5.0:
            break
            
        # Adjust beta based on iteration? Start high for sharpness, lower for exploration?
        # Let's try a few betas for the first few configs
        beta = 100.0 if i < 5 else 50.0
        
        try:
            opt_ang, score = _optimize_angles(init_ang, time_per_config, beta)
            if score > best_score:
                best_score = score
                best_angles = opt_ang
        except Exception as e:
            pass
            
    # If no good result, fallback to regular polygon
    if best_score < 0.1:
        best_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_score = _score_angles(best_angles)
        
    # Final polish with higher beta on the best angles
    if best_angles is not None and time.time() < deadline - 2.0:
        final_ang, final_score = _optimize_angles(best_angles, deadline - time.time() - 2.0, beta=200.0)
        if final_score > best_score:
            best_score = final_score
            best_angles = final_ang
            
    # Convert angles to points
    if best_angles is None:
        return np.random.rand(N, 2)
        
    points = _get_points_from_angles(best_angles)
    return points
