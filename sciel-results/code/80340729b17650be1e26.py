import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 140.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _greedy_init(rng):
    """
    Greedy initialization: place points one by one to maximize the minimum distance to existing points.
    """
    pts = []
    p0 = _random_point_in_triangle(rng)
    pts.append(np.array(p0))
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        for _ in range(200):
            cand = np.array(_random_point_in_triangle(rng))
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def _soft_min_area(pts, beta=1000.0):
    """
    Compute a smooth approximation of the minimum triangle area.
    Uses the log-sum-exp trick: -1/beta * log(sum(exp(-beta * area))) approximates min(area).
    This is differentiable, allowing L-BFGS-B to work effectively.
    """
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # To avoid numerical issues, shift areas
    max_area = np.max(areas)
    exp_args = -beta * (areas - max_area)
    # Clip to prevent underflow/overflow in exp, though log-sum-exp is stable if done right
    # sum(exp(exp_args)) is sum of terms <= 1
    sum_exp = np.sum(np.exp(exp_args))
    
    if sum_exp == 0:
        return max_area
    
    soft_min = max_area - (1.0 / beta) * np.log(sum_exp)
    
    return soft_min

def _objective_and_grad(pts_flat, beta=1000.0):
    """
    Objective function and its gradient for L-BFGS-B.
    We want to maximize the soft-min area.
    So we minimize -soft_min_area.
    """
    pts = pts_flat.reshape(N, 2)
    
    # Calculate areas
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Gradient of soft-min
    # soft_min = max_a - 1/beta * log(sum(exp(-beta*(a - max_a))))
    # Let z_i = -beta * (a_i - max_a)
    # sum_exp = sum(exp(z_i))
    # soft_min = max_a - 1/beta * log(sum_exp)
    
    # Note: max_a is one of the areas. The derivative of max_a w.r.t pts is non-smooth (0 for all except the max, where it's gradient of that area).
    # However, the log-sum-exp form is smooth.
    # Let's compute gradient directly from the formula:
    # f = - (1/beta) * log( sum_i exp(-beta * a_i) ) + max_a
    # Wait, the standard soft min is:
    # m_beta(a) = -1/beta * log( sum_i exp(-beta * a_i) )
    # This is smooth. The + max_a term in my previous derivation was for numerical stability but changes the function slightly?
    # Actually, -1/beta * log(sum(exp(-beta*(a_i - M)))) = -1/beta * log(exp(-beta*M) * sum(exp(-beta*a_i)))
    # = -1/beta * (-beta*M + log(sum(exp(-beta*a_i))))
    # = M - 1/beta * log(sum(exp(-beta*a_i)))
    # So yes, it is M + soft_min_original.
    # The gradient of M (max area) is tricky to compute analytically in a vectorized way without knowing which is max.
    # But the gradient of the log-sum-exp part is easy.
    # grad(m_beta) = grad(M) - 1/beta * grad(log(sum(exp(-beta*a_i))))
    # grad(log(sum(exp(-beta*a_i)))) = sum(exp(-beta*a_i) * (-beta * grad(a_i))) / sum(exp(-beta*a_i))
    # = -beta * sum(w_i * grad(a_i)) where w_i = exp(-beta*a_i) / sum(exp(-beta*a_i))
    # So grad(m_beta) = grad(M) + sum(w_i * grad(a_i))
    
    # As beta -> infinity, w_i concentrates on the minimum area index k*.
    # grad(M) is the gradient of the area at the maximum index.
    # This hybrid makes it hard to get a clean analytic gradient for the "M" part if we don't track indices.
    
    # Alternative: Just use the standard soft-min without the +M shift for the gradient calculation, 
    # and rely on the fact that for large beta it approximates min.
    # m(a) = -1/beta * log(sum(exp(-beta * a_i)))
    # grad m = -1/beta * (sum(exp(-beta*a_i)*(-beta*grad a_i)) / sum(exp(-beta*a_i)))
    # grad m = sum( (exp(-beta*a_i)/sum(exp(-beta*a_j))) * grad a_i )
    # This is a convex combination of the gradients of the individual areas.
    # This is perfectly smooth and easy to compute.
    # The value m(a) approximates min(a) from above.
    # We want to maximize m(a).
    
    # Let's compute weights w_i
    # To avoid underflow, subtract max exponent (which is -beta * min(area))
    min_area_val = np.min(areas)
    exp_args = -beta * (areas - min_area_val)
    # exp_args <= 0. Max is 0.
    exp_vals = np.exp(exp_args)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        # Fallback
        val = min_area_val
        grad = np.zeros_like(pts_flat)
        return -val, -grad
        
    weights = exp_vals / sum_exp
    
    # Value
    val = min_area_val - (1.0 / beta) * np.log(sum_exp)
    
    # Gradient of val w.r.t. pts
    # grad(val) = sum_i weights_i * grad(areas_i)
    
    # grad(areas_i) w.r.t. pts
    # area_i = 0.5 * | (p_j - p_i) x (p_k - p_i) |
    # Let cross_i = (p_j - p_i) x (p_k - p_i)
    # area_i = 0.5 * |cross_i|
    # grad(area_i) = 0.5 * sign(cross_i) * grad(cross_i)
    
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # d(cross)/d(x_i) = -(y_k - y_i) - (y_j - y_i)*(-1) = -y_k + y_i + y_j - y_i = y_j - y_k
    # d(cross)/d(y_i) = -(x_k - x_i) - (x_j - x_i)*(-1) = -x_k + x_i + x_j - x_i = x_j - x_k
    # d(cross)/d(x_j) = (y_k - y_i)
    # d(cross)/d(y_j) = -(x_k - x_i)
    # d(cross)/d(x_k) = -(y_j - y_i)
    # d(cross)/d(y_k) = (x_j - x_i)
    
    # Let's compute these vectors for all triplets
    # p_i, p_j, p_k are (N_triplets, 2)
    
    # For point i in triplet t:
    # d_area/d_x_i = 0.5 * sign(cross) * (y_j - y_k)
    # d_area/d_y_i = 0.5 * sign(cross) * (x_j - x_k)
    
    # For point j in triplet t:
    # d_area/d_x_j = 0.5 * sign(cross) * (y_k - y_i)
    # d_area/d_y_j = 0.5 * sign(cross) * (-(x_k - x_i))
    
    # For point k in triplet t:
    # d_area/d_x_k = 0.5 * sign(cross) * (-(y_j - y_i))
    # d_area/d_y_k = 0.5 * sign(cross) * (x_j - x_i)
    
    sign_cross = np.sign(cross)
    # If cross is 0, gradient is 0 (or undefined, but 0 is safe)
    half_sign = 0.5 * sign_cross
    
    # We need to accumulate gradients for each point index.
    # grad_pts is (N, 2)
    grad_pts = np.zeros((N, 2))
    
    # Contributions from i
    # dx_i = half_sign * (p_j[:,1] - p_k[:,1])
    # dy_i = half_sign * (p_j[:,0] - p_k[:,0])
    dx_i = half_sign * (p_j[:, 1] - p_k[:, 1])
    dy_i = half_sign * (p_j[:, 0] - p_k[:, 0])
    
    # Contributions from j
    dx_j = half_sign * (p_k[:, 1] - p_i[:, 1])
    dy_j = half_sign * (-(p_k[:, 0] - p_i[:, 0]))
    
    # Contributions from k
    dx_k = half_sign * (-(p_j[:, 1] - p_i[:, 1]))
    dy_k = half_sign * (p_j[:, 0] - p_i[:, 0])
    
    # Weighted sum
    # We need to sum w_t * dx_i_t for all t where i is the first index, etc.
    # This is a scatter-add operation.
    
    # Flatten indices for np.add.at
    idx_i = _i_idx
    idx_j = _j_idx
    idx_k = _k_idx
    
    # Accumulate x and y gradients
    # x-gradient
    np.add.at(grad_pts, (idx_i, 0), weights * dx_i)
    np.add.at(grad_pts, (idx_j, 0), weights * dx_j)
    np.add.at(grad_pts, (idx_k, 0), weights * dx_k)
    
    # y-gradient
    np.add.at(grad_pts, (idx_i, 1), weights * dy_i)
    np.add.at(grad_pts, (idx_j, 1), weights * dy_j)
    np.add.at(grad_pts, (idx_k, 1), weights * dy_k)
    
    # We want to MINIMIZE -val
    # So objective = -val
    # Gradient = -grad_pts
    
    obj = -val
    grad = -grad_pts.flatten()
    
    return obj, grad

def _penalty_objective_and_grad(pts_flat, beta=1000.0, lambda_pen=100.0):
    """
    Objective with penalty for points outside the triangle.
    """
    obj_area, grad_area = _objective_and_grad(pts_flat, beta)
    
    pts = pts_flat.reshape(N, 2)
    
    # Penalty for being outside
    # Distance to boundaries:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x  =>  sqrt(3)*x - y >= 0
    # 3. y <= sqrt(3) * (1-x) => sqrt(3)*(1-x) - y >= 0
    
    d1 = pts[:, 1]
    d2 = np.sqrt(3) * pts[:, 0] - pts[:, 1]
    d3 = np.sqrt(3) * (1 - pts[:, 0]) - pts[:, 1]
    
    # Penalty is sum of max(0, -d)^2
    # We use a smooth approximation or just quadratic penalty on negative parts?
    # Simple quadratic penalty on violation:
    # viol = max(0, -d)
    # pen = sum(viol^2)
    # grad_pen = sum(2 * viol * grad(viol))
    # grad(viol) = -1 if d < 0, else 0
    
    viol1 = np.maximum(0, -d1)
    viol2 = np.maximum(0, -d2)
    viol3 = np.maximum(0, -d3)
    
    pen = np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2)
    
    grad_pen = np.zeros_like(pts)
    
    mask1 = d1 < 0
    grad_pen[mask1, 1] += 2 * viol1[mask1]
    
    mask2 = d2 < 0
    # d2 = sqrt(3)*x - y
    # grad d2 = [sqrt(3), -1]
    # grad viol2 = -grad d2 = [-sqrt(3), 1]
    # grad pen = 2 * viol * grad viol
    grad_pen[mask2, 0] += 2 * viol2[mask2] * (-np.sqrt(3))
    grad_pen[mask2, 1] += 2 * viol2[mask2] * (1)
    
    mask3 = d3 < 0
    # d3 = sqrt(3)*(1-x) - y = sqrt(3) - sqrt(3)*x - y
    # grad d3 = [-sqrt(3), -1]
    # grad viol3 = -grad d3 = [sqrt(3), 1]
    grad_pen[mask3, 0] += 2 * viol3[mask3] * (np.sqrt(3))
    grad_pen[mask3, 1] += 2 * viol3[mask3] * (1)
    
    obj = obj_area + lambda_pen * pen
    grad = grad_area + lambda_pen * grad_pen.flatten()
    
    return obj, grad

def _objective_wrapper(pts_flat, beta, lambda_pen):
    obj, _ = _penalty_objective_and_grad(pts_flat, beta, lambda_pen)
    return obj

def _grad_wrapper(pts_flat, beta, lambda_pen):
    _, grad = _penalty_objective_and_grad(pts_flat, beta, lambda_pen)
    return grad

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    initializations = []
    
    # 1. Greedy
    try:
        p_greedy = _greedy_init(rng)
        initializations.append(p_greedy)
    except Exception:
        pass
        
    # 2. Random
    p_rand = np.array([_random_point_in_triangle(rng) for _ in range(N)])
    p_rand = _project_to_triangle(p_rand)
    initializations.append(p_rand)
    
    # 3. Structured
    p_struct = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    p_struct = np.array(p_struct)
    p_struct = _project_to_triangle(p_struct)
    initializations.append(p_struct)
    
    # 4. More random starts
    for _ in range(5):
        p_r = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        p_r = _project_to_triangle(p_r)
        initializations.append(p_r)
        
    # Optimize each start
    # We use a schedule for beta and lambda_pen
    # Start with lower beta (smoother) and higher penalty, then increase beta (sharper) and lower penalty
    
    for s, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 15:
            break
            
        x0 = init_pts.flatten()
        
        # Schedule
        # Phase 1: Smooth, high penalty
        beta_1 = 50.0
        lam_1 = 1000.0
        # Phase 2: Sharper, lower penalty
        beta_2 = 500.0
        lam_2 = 100.0
        # Phase 3: Very sharp, low penalty
        beta_3 = 2000.0
        lam_3 = 10.0
        
        current_x = x0
        
        for beta, lam in [(beta_1, lam_1), (beta_2, lam_2), (beta_3, lam_3)]:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            try:
                result = minimize(
                    _objective_wrapper,
                    current_x,
                    jac=_grad_wrapper,
                    args=(beta, lam),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_x = result.x
            except Exception:
                pass
        
        # Final evaluation
        pts_opt = _project_to_triangle(current_x.reshape(N, 2))
        
        if _check_inside(pts_opt):
            score = _min_triangle_area(pts_opt)
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
                
    # If no optimization succeeded, fall back to the best initial guess
    if best_pts is None:
        for init_pts in initializations:
            score = _min_triangle_area(init_pts)
            if score > best_score:
                best_score = score
                best_pts = init_pts.copy()
                
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
        
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
