import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])
NUM_TRIPLES = TRIPLE_INDICES.shape[0]

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    # Calculate Minimum Triangle Area
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _soft_min_score(points, temp=0.05):
    """
    Differentiable approximation of min triangle area / hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -T * log(sum(exp(-areas/T)))
    # To avoid overflow, we can subtract max, but for T=0.05 and areas ~ 0.1, exp is fine.
    # However, if areas are large, exp(-large) is small.
    # If areas are very small, exp(-small) is ~1.
    # Let's use the stable form: max_area - T * log(sum(exp(-(areas - max_area)/T)))
    max_area = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-(areas - max_area) / temp)))
    soft_min_area = max_area - temp * log_sum
    
    return soft_min_area / hull_area

def _soft_min_score_and_grad(points, temp=0.05):
    """
    Calculate soft-min score and its gradient w.r.t. coordinates.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0, np.zeros_like(points)
    except Exception:
        return 0.0, np.zeros_like(points)
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    # Cross product vector: c = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
    dx_ji = p_j[:, 0] - p_i[:, 0]
    dy_ji = p_j[:, 1] - p_i[:, 1]
    dx_ki = p_k[:, 0] - p_i[:, 0]
    dy_ki = p_k[:, 1] - p_i[:, 1]
    
    cross = dx_ji * dy_ki - dy_ji * dx_ki
    areas = 0.5 * np.abs(cross)
    
    # Weights for soft-min
    # w_i = exp(-A_i / T) / sum(exp(-A_j / T))
    # To be stable:
    max_area = np.max(areas)
    exp_terms = np.exp(-(areas - max_area) / temp)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp # Shape (NUM_TRIPLES,)
    
    soft_min_area = max_area - temp * np.log(sum_exp)
    score = soft_min_area / hull_area
    
    # Gradient of soft_min_area w.r.t. coordinates
    # d(soft_min_area)/dx = sum_i weights_i * d(A_i)/dx
    # Note: A_i = 0.5 * |cross_i|. 
    # If cross_i > 0, dA_i/dx = 0.5 * dcross_i/dx.
    # If cross_i < 0, dA_i/dx = -0.5 * dcross_i/dx.
    # So dA_i/dx = 0.5 * sign(cross_i) * dcross_i/dx.
    
    signs = np.sign(cross)
    # If cross is 0, sign is 0, gradient is 0 (which is correct for |x| at 0, subgradient, but 0 is a valid choice for optimization).
    
    grad_area = np.zeros_like(points)
    
    # dcross/dx for point i, j, k in triple t
    # cross = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
    # d/dxi = -(yk-yi) + (yj-yi) = yi - yk
    # d/dyi = -(xj-xi) + (xk-xi) = xi - xj
    # d/dxj = (yk-yi)
    # d/dyj = -(xk-xi)
    # d/dxk = -(yj-yi)
    # d/dyk = (xj-xi)
    
    # We accumulate contributions from all triples.
    # For each triple t:
    #   pt_i, pt_j, pt_k
    #   contrib = 0.5 * signs[t] * weights[t] * [grad_x, grad_y]
    
    # Vectorized accumulation is tricky with indices, so we loop over triples 
    # but N=14, 364 triples. 364 iterations is fast.
    
    for idx in range(NUM_TRIPLES):
        w = 0.5 * signs[idx] * weights[idx]
        if w == 0: continue
        
        i, j, k = TRIPLE_INDICES[idx]
        
        # Gradients for point i
        grad_i_x = (points[i, 1] - points[k, 1])
        grad_i_y = (points[i, 0] - points[j, 0])
        
        # Gradients for point j
        grad_j_x = (points[k, 1] - points[i, 1])
        grad_j_y = (points[i, 0] - points[k, 0])
        
        # Gradients for point k
        grad_k_x = (points[i, 1] - points[j, 1])
        grad_k_y = (points[j, 0] - points[i, 0])
        
        grad_area[i, 0] += w * grad_i_x
        grad_area[i, 1] += w * grad_i_y
        grad_area[j, 0] += w * grad_j_x
        grad_area[j, 1] += w * grad_j_y
        grad_area[k, 0] += w * grad_k_x
        grad_area[k, 1] += w * grad_k_y
        
    # Chain rule for score = soft_min_area / hull_area
    # d(score)/dx = (hull_area * d(soft_min_area)/dx - soft_min_area * d(hull_area)/dx) / hull_area^2
    # Approximating d(hull_area)/dx as 0 is common in these problems because the hull area changes slowly 
    # and the min area is the bottleneck. However, for better accuracy, we could estimate it.
    # Given the complexity, we will assume d(hull_area)/dx is negligible or use a finite difference if needed.
    # For now, let's just use the gradient of the numerator, scaled by 1/hull_area.
    # This is an approximation: grad_score ~= grad_soft_min_area / hull_area
    
    grad_score = grad_area / hull_area
    
    return score, grad_score

def _generate_two_ring_points(r_inner, phase):
    """
    Generate points for two-ring configuration.
    Outer: 7 points at radius 1.
    Inner: 7 points at radius r_inner, rotated by phase.
    """
    angles_outer = 2 * np.pi * np.arange(7) / 7
    angles_inner = angles_outer + phase
    
    x_outer = np.cos(angles_outer)
    y_outer = np.sin(angles_outer)
    
    x_inner = r_inner * np.cos(angles_inner)
    y_inner = r_inner * np.sin(angles_inner)
    
    # Interleave to maintain some order? Or just stack?
    # The score doesn't depend on order, but for polish, order doesn't matter.
    points = np.vstack([
        np.column_stack((x_outer, y_outer)),
        np.column_stack((x_inner, y_inner))
    ])
    return points

def _optimize_two_ring(time_limit):
    """
    Optimize the two-ring parameters using DE.
    """
    start_time = time.time()
    
    def obj(params):
        r_inner, phase = params
        if r_inner < 0.05 or r_inner > 0.95:
            return 100.0
        points = _generate_two_ring_points(r_inner, phase)
        return -_soft_min_score(points, temp=0.05)
        
    bounds = [(0.1, 0.9), (0, 2*np.pi/7)]
    
    try:
        res = differential_evolution(
            obj,
            bounds,
            seed=42,
            maxiter=30,
            popsize=20,
            tol=1e-6,
            polish=False
        )
        if time.time() - start_time < time_limit:
            best_params = res.x
            best_score = -res.fun
            points = _generate_two_ring_points(best_params[0], best_params[1])
            return points, best_score
    except Exception:
        pass
        
    # Fallback
    points = _generate_two_ring_points(0.5, np.pi/7)
    return points, _soft_min_score(points)

def _optimize_fourier(time_limit):
    """
    Optimize Fourier radius distortion.
    """
    start_time = time.time()
    
    def _fourier_points(params):
        K = (len(params)) // 2
        n = np.arange(N)
        angles = 2 * np.pi * n / N
        r = np.ones(N)
        
        for k in range(1, K+1):
            a_k = params[2*(k-1)]
            b_k = params[2*(k-1)+1]
            r += a_k * np.cos(2 * np.pi * k * n / N) + b_k * np.sin(2 * np.pi * k * n / N)
            
        r = np.maximum(r, 0.5)
        
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        
        return np.column_stack((x, y))
        
    def obj(params):
        points = _fourier_points(params)
        return -_soft_min_score(points, temp=0.05)
        
    K = 3
    n_params = 2 * K
    bounds = [(-0.3, 0.3)] * n_params
    
    try:
        res = differential_evolution(
            obj,
            bounds,
            seed=123,
            maxiter=30,
            popsize=20,
            tol=1e-6,
            polish=False
        )
        if time.time() - start_time < time_limit:
            points = _fourier_points(res.x)
            return points, -res.fun
    except Exception:
        pass
        
    # Fallback
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    return points, _soft_min_score(points)

def _coordinate_polish(points, time_limit):
    """
    Polish coordinates using L-BFGS-B with soft-min gradient, then Nelder-Mead with hard score.
    """
    start_time = time.time()
    current_points = points.copy()
    
    # Phase 1: L-BFGS-B with Soft-Min
    def soft_obj_and_grad(flat_params):
        pts = flat_params.reshape(-1, 2)
        score, grad = _soft_min_score_and_grad(pts, temp=0.05)
        return -score, -grad.flatten()
        
    x0 = current_points.flatten()
    
    try:
        res = minimize(
            soft_obj_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.success or -res.fun > -soft_obj_and_grad(x0)[0]:
            current_points = res.x.reshape(-1, 2)
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.7:
        return current_points
        
    # Phase 2: Nelder-Mead with Hard Score
    def hard_obj(flat_params):
        pts = flat_params.reshape(-1, 2)
        return -_calculate_score(pts)
        
    x0 = current_points.flatten()
    initial_score = hard_obj(x0)
    
    try:
        res = minimize(
            hard_obj,
            x0,
            method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000, 'maxfev': 5000}
        )
        if -res.fun > initial_score:
            current_points = res.x.reshape(-1, 2)
    except Exception:
        pass
        
    return current_points

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Two-Ring Optimization
    # This is a strong candidate for high symmetry solutions.
    score1, points1 = _optimize_two_ring(40.0)
    if score1 > best_score:
        best_score = score1
        best_points = points1
        
    # Strategy 2: Fourier Optimization
    if time.time() < deadline - 30:
        score2, points2 = _optimize_fourier(30.0)
        if score2 > best_score:
            best_score = score2
            best_points = points2
            
    # Fallback to Regular Polygon if nothing found
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _calculate_score(best_points)
        
    # Final Coordinate Polish
    if time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 2
        best_points = _coordinate_polish(best_points, remaining_time)
        final_score = _calculate_score(best_points)
        if final_score > best_score:
            best_score = final_score
            
    return best_points
