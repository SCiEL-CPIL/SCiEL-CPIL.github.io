"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Use a smooth surrogate objective (log-sum-exp) to make the min-area differentiable.
2. Optimize with L-BFGS-B on barycentric coordinates (enforcing triangle constraint exactly).
3. Multi-start with diverse structured and random initializations.
4. Refine the best solutions using a direct min-area local search (Nelder-Mead) to escape
   the smoothing bias and find the true min-area maximizer.
5. Use time-budgeted iterations to fully utilize the 120s limit.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 115.0  # Safety margin before 150s limit

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

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian coordinates.
    bary: shape (N, 3) where each row sums to 1 and all elements >= 0.
    Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    """
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack((x, y))

def _cartesian_to_bary(pts):
    """Convert Cartesian points to barycentric coordinates relative to A(0,0), B(1,0), C(0.5, S3H)."""
    b3 = pts[:, 1] / S3H
    b2 = pts[:, 0] - 0.5 * b3
    b1 = 1.0 - b2 - b3
    return np.column_stack((b1, b2, b3))

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

def _soft_min_objective(bary, beta):
    """
    Smooth approximation of the negative minimum triangle area using log-sum-exp.
    bary: shape (N, 3)
    beta: temperature parameter (larger = sharper min)
    Returns: -soft_min(areas) and gradient w.r.t. bary
    """
    pts = _barycentric_to_cartesian(bary)
    
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min using log-sum-exp: soft_min(a) = -1/beta * log(sum(exp(-beta * a)))
    # For numerical stability:
    max_a = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_a))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = -1.0 / beta * log_sum + max_a
    
    # Gradient of soft_min w.r.t. areas:
    # d(soft_min)/d(a_k) = -exp(-beta*(a_k - max_a)) / sum(exp(-beta*(a_k - max_a)))
    # = -exp_terms / sum(exp_terms)
    grad_areas = -exp_terms / np.sum(exp_terms)
    
    # Now we need gradient w.r.t. barycentric coordinates.
    # This requires chain rule: d(soft_min)/d(bary) = sum_k grad_areas[k] * d(area_k)/d(bary)
    # area_k = 0.5 * |cross_k|
    # cross_k = (p_j - p_i) x (p_k - p_i)
    # This is complex. For L-BFGS-B, we can use finite differences or provide a sparse gradient.
    # For simplicity and speed, let's use finite differences via L-BFGS-B's default.
    
    return -soft_min_val

def _soft_min_grad(bary, beta):
    """
    Gradient of the soft min objective w.r.t. barycentric coordinates.
    This is expensive to compute analytically, so we use finite differences.
    However, for L-BFGS-B, we can just return None and let it use FD.
    But FD is slow. Let's implement a simple FD gradient.
    """
    eps = 1e-7
    grad = np.zeros_like(bary)
    f0 = _soft_min_objective(bary, beta)
    
    for i in range(N):
        for j in range(3):
            bary_plus = bary.copy()
            bary_plus[i, j] += eps
            # Renormalize to keep sum=1 and non-negative (approximate)
            bary_plus[i] = np.maximum(bary_plus[i], 0)
            s = np.sum(bary_plus[i])
            if s > 0:
                bary_plus[i] /= s
            f_plus = _soft_min_objective(bary_plus, beta)
            grad[i, j] = (f_plus - f0) / eps
    
    return grad

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
        for _ in range(500):
            cand = np.array(_random_point_in_triangle(rng))
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def _structured_init():
    """Create a structured initialization based on known Heilbronn-like configurations."""
    # Use a hexagonal-like pattern scaled to the triangle
    # Vertices, edge midpoints, and interior points
    pts = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    return np.array(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses L-BFGS-B with soft-min objective for global exploration,
    then Nelder-Mead for local refinement on the true min-area objective.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Multi-start optimization
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
    p_struct = _structured_init()
    p_struct = _project_to_triangle(p_struct)
    initializations.append(p_struct)
    
    # 4. More random starts
    for _ in range(5):
        p_rand2 = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        p_rand2 = _project_to_triangle(p_rand2)
        initializations.append(p_rand2)
    
    # Phase 1: L-BFGS-B with soft-min objective
    # Use a range of beta values for annealing effect
    betas = [10.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        for s, init_pts in enumerate(initializations):
            if time.time() - start_time > DEADLINE - 20:
                break
            
            bary = _cartesian_to_bary(init_pts)
            bary = np.maximum(bary, 0)
            bary = bary / np.sum(bary, axis=1, keepdims=True)
            
            def obj_func(bary_flat):
                bary_tmp = bary_flat.reshape(N, 3)
                bary_tmp = np.maximum(bary_tmp, 0)
                bary_tmp = bary_tmp / np.sum(bary_tmp, axis=1, keepdims=True)
                return _soft_min_objective(bary_tmp, beta)
            
            def grad_func(bary_flat):
                bary_tmp = bary_flat.reshape(N, 3)
                bary_tmp = np.maximum(bary_tmp, 0)
                bary_tmp = bary_tmp / np.sum(bary_tmp, axis=1, keepdims=True)
                return _soft_min_grad(bary_tmp, beta).flatten()
            
            bounds = [(0, 1)] * (N * 3)
            
            try:
                result = minimize(
                    obj_func,
                    bary.flatten(),
                    jac=grad_func,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                if result.success:
                    bary_opt = result.x.reshape(N, 3)
                    bary_opt = np.maximum(bary_opt, 0)
                    bary_opt = bary_opt / np.sum(bary_opt, axis=1, keepdims=True)
                    pts_opt = _barycentric_to_cartesian(bary_opt)
                    pts_opt = _project_to_triangle(pts_opt)
                    
                    if _check_inside(pts_opt):
                        score = _min_triangle_area(pts_opt)
                        if score > best_score:
                            best_score = score
                            best_pts = pts_opt.copy()
                            
            except Exception as e:
                pass
    
    # Phase 2: Nelder-Mead refinement on the true min-area objective
    if best_pts is not None:
        if time.time() - start_time < DEADLINE - 10:
            def nm_obj(pts_flat):
                pts_tmp = pts_flat.reshape(N, 2)
                pts_tmp = _project_to_triangle(pts_tmp)
                return -_min_triangle_area(pts_tmp)
            
            try:
                result = minimize(
                    nm_obj,
                    best_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                
                if result.success:
                    pts_opt = result.x.reshape(N, 2)
                    pts_opt = _project_to_triangle(pts_opt)
                    
                    if _check_inside(pts_opt):
                        score = _min_triangle_area(pts_opt)
                        if score > best_score:
                            best_score = score
                            best_pts = pts_opt.copy()
                            
            except Exception as e:
                pass
    
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
