"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Barycentric coordinate parameterization to enforce the triangle constraint exactly.
2. Smooth approximation of the minimum area using log-sum-exp to create a differentiable objective.
3. Multi-start optimization using L-BFGS-B on the smooth objective.
4. Refine the best solution using a more accurate (less smooth) objective or direct evaluation.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
# This is done once at module load time for efficiency
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
    # pts shape (N, 2)
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

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian coordinates.
    bary: shape (N, 3) where each row sums to 1 and all elements >= 0.
    Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    """
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * S3H
    return np.column_stack((x, y))

def _soft_min_objective(x_flat, beta=50.0):
    """
    Smooth approximation of the negative minimum triangle area.
    x_flat: shape (N, 2) flattened.
    We parameterize points in Cartesian coordinates but penalize points outside the triangle.
    Alternatively, we could parameterize in barycentric coordinates (log-sum-exp trick for simplex).
    However, L-BFGS-B works well with Cartesian + penalty if the penalty is strong enough.
    Let's stick to Cartesian with a strong quadratic penalty for being outside.
    """
    pts = x_flat.reshape(N, 2)
    
    # Penalty for points outside the triangle
    penalty = 0.0
    for p in pts:
        x, y = p
        d1 = y
        d2 = np.sqrt(3) * x - y
        d3 = np.sqrt(3) * (1 - x) - y
        if d1 < 0: penalty += d1**2
        if d2 < 0: penalty += d2**2
        if d3 < 0: penalty += d3**2
    
    # Compute areas
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -min(a) ~ (1/beta) * log(sum(exp(-beta * a)))
    # We want to maximize min(area), so we minimize -soft_min(area).
    # soft_min = (1/beta) * log(sum(exp(-beta * areas)))
    # obj = -soft_min + penalty
    
    # To avoid numerical issues with large exponentials, subtract the max (which is -min)
    # Actually, for log-sum-exp stability:
    # log(sum(exp(x_i))) = max(x_i) + log(sum(exp(x_i - max(x_i))))
    # Here x_i = -beta * areas.
    # max(x_i) = -beta * min(areas).
    # So log(sum(exp(-beta * areas))) = -beta * min(areas) + log(sum(exp(-beta * (areas - min(areas)))))
    # This is stable.
    
    min_area = np.min(areas)
    exp_terms = np.exp(-beta * (areas - min_area))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = min_area + (1.0 / beta) * log_sum
    
    obj = -soft_min_val + 1000.0 * penalty
    
    return obj

def _soft_min_grad(x_flat, beta=50.0):
    """Gradient of the soft min objective."""
    pts = x_flat.reshape(N, 2)
    
    # Penalty gradient
    grad_pen = np.zeros_like(pts)
    for i, p in enumerate(pts):
        x, y = p
        d1 = y
        d2 = np.sqrt(3) * x - y
        d3 = np.sqrt(3) * (1 - x) - y
        
        if d1 < 0:
            grad_pen[i, 1] += 2.0 * d1
        if d2 < 0:
            grad_pen[i, 0] += 2.0 * np.sqrt(3) * d2
            grad_pen[i, 1] += -2.0 * d2
        if d3 < 0:
            grad_pen[i, 0] += -2.0 * np.sqrt(3) * d3
            grad_pen[i, 1] += -2.0 * d3
    
    # Area gradient
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    exp_terms = np.exp(-beta * (areas - min_area))
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient of -soft_min = -d(soft_min)/d(pts)
    # soft_min = min_area + (1/beta) * log(sum(exp(-beta * (areas - min_area))))
    # d(soft_min)/d(area_m) = weights[m]
    # d(area_m)/d(pts) depends on the triplet m.
    
    # This is complex to compute analytically for all points.
    # For simplicity, we can use numerical gradient if analytic is too slow/complex,
    # or just use L-BFGS-B with finite differences (default).
    # Given the time limit, let's rely on L-BFGS-B's internal finite difference or
    # implement a simpler gradient.
    
    # Actually, let's just return the penalty gradient and let L-BFGS-B handle the rest?
    # No, L-BFGS-B needs the full gradient.
    # Let's skip the analytic gradient for the area part and use finite differences?
    # Or better, implement the gradient properly.
    
    # d(area_m)/d(p_i) = 0.5 * (dy2 - dy1) ? No.
    # area = 0.5 * |cross|. Let's assume cross > 0 for simplicity (orientation).
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # d(cross)/d(x_i) = -(y_k - y_i) + (y_j - y_i) = y_j - y_k
    # d(cross)/d(y_i) = -(x_k - x_i) + (x_j - x_i) = x_j - x_k
    # d(cross)/d(x_j) = (y_k - y_i)
    # d(cross)/d(y_j) = -(x_k - x_i)
    # d(cross)/d(x_k) = -(y_j - y_i)
    # d(cross)/d(y_k) = (x_j - x_i)
    
    # Since we take abs, the sign of cross matters.
    # Let's compute the gradient of -min(areas) using the weights.
    
    grad_area = np.zeros_like(pts)
    
    for m in range(len(_triplets)):
        w = weights[m]
        if w < 1e-10:
            continue
            
        i, j, k = _triplets[m]
        sign = 1.0 if cross[m] >= 0 else -1.0
        
        # d(0.5 * cross)/d(pts)
        # d(cross)/d(x_i) = y_j - y_k
        # d(cross)/d(y_i) = x_j - x_k
        # d(cross)/d(x_j) = y_k - y_i
        # d(cross)/d(y_j) = x_i - x_k
        # d(cross)/d(x_k) = y_i - y_j
        # d(cross)/d(y_k) = x_i - x_j
        
        grad_area[i, 0] += 0.5 * sign * (pts[j, 1] - pts[k, 1]) * w
        grad_area[i, 1] += 0.5 * sign * (pts[j, 0] - pts[k, 0]) * w
        
        grad_area[j, 0] += 0.5 * sign * (pts[k, 1] - pts[i, 1]) * w
        grad_area[j, 1] += 0.5 * sign * (pts[i, 0] - pts[k, 0]) * w
        
        grad_area[k, 0] += 0.5 * sign * (pts[i, 1] - pts[j, 1]) * w
        grad_area[k, 1] += 0.5 * sign * (pts[i, 0] - pts[j, 0]) * w
        
    # Objective is -soft_min + penalty
    # d(-soft_min)/d(pts) = - (1/beta) * sum(weights * d(areas)/d(pts)) ?
    # Wait, soft_min = min_area + (1/beta) * log_sum
    # d(soft_min)/d(pts) = (1/beta) * sum_m (weights_m * d(area_m)/d(pts))
    # So d(-soft_min)/d(pts) = -(1/beta) * sum_m (weights_m * d(area_m)/d(pts))
    
    # My grad_area above is sum_m (weights_m * d(area_m)/d(pts))
    # So the gradient of the area part of the objective is:
    grad_obj_area = -(1.0 / beta) * grad_area
    
    total_grad = grad_obj_area + 1000.0 * grad_pen
    
    return total_grad.flatten()

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

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Multi-start L-BFGS-B on a smooth objective.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Multi-start optimization
    n_starts = 20
    
    for s in range(n_starts):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Generate initial guess
        # Mix of random and structured
        if s == 0:
            # Greedy-like init
            pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
            pts = _project_to_triangle(pts)
        else:
            # Random init
            pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
            pts = _project_to_triangle(pts)
        
        # Optimize
        try:
            result = minimize(
                _soft_min_objective,
                pts.flatten(),
                jac=_soft_min_grad,
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 100}
            )
            optimized_pts = result.x.reshape(N, 2)
            optimized_pts = _project_to_triangle(optimized_pts)
            
            if _check_inside(optimized_pts):
                score = _min_triangle_area(optimized_pts)
                if score > best_score:
                    best_score = score
                    best_pts = optimized_pts.copy()
        except Exception:
            pass
    
    # Final polish with higher beta (sharper min)
    if best_pts is not None and time.time() - start_time < DEADLINE - 2:
        # Increase beta for sharper approximation
        # We need to modify the objective to use a different beta.
        # Let's just run another round with a fixed higher beta.
        
        def soft_min_obj_high_beta(x_flat, beta=200.0):
            pts = x_flat.reshape(N, 2)
            
            penalty = 0.0
            for p in pts:
                x, y = p
                d1 = y
                d2 = np.sqrt(3) * x - y
                d3 = np.sqrt(3) * (1 - x) - y
                if d1 < 0: penalty += d1**2
                if d2 < 0: penalty += d2**2
                if d3 < 0: penalty += d3**2
            
            p_i = pts[_i_idx]
            p_j = pts[_j_idx]
            p_k = pts[_k_idx]
            
            dx1 = p_j - p_i
            dx2 = p_k - p_i
            
            cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
            areas = 0.5 * np.abs(cross)
            
            min_area = np.min(areas)
            exp_terms = np.exp(-beta * (areas - min_area))
            log_sum = np.log(np.sum(exp_terms))
            soft_min_val = min_area + (1.0 / beta) * log_sum
            
            obj = -soft_min_val + 1000.0 * penalty
            return obj
        
        try:
            result = minimize(
                lambda x: soft_min_obj_high_beta(x),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 100}
            )
            polished_pts = result.x.reshape(N, 2)
            polished_pts = _project_to_triangle(polished_pts)
            
            if _check_inside(polished_pts):
                score = _min_triangle_area(polished_pts)
                if score > best_score:
                    best_score = score
                    best_pts = polished_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
        
    return best_pts
