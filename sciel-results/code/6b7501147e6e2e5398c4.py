"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 145.0  # Safety margin before 150s kill

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(points):
    """
    Return array of all triangle areas (N choose 3) for N points.
    Vectorized inner loop for speed.
    """
    n = len(points)
    if n < 3:
        return np.array([0.0])
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Points k > j
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            # Cross product of (p_j - p_i) and (p_k - p_i)
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _min_triangle_area(points):
    """Compute the minimum triangle area."""
    areas = _all_triangle_areas(points)
    return np.min(areas) if len(areas) > 0 else 0.0

def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    if min_area <= 1e-12:
        return 0.0
    return min_area / hull_a

def _soft_min_and_grad(pts, beta):
    """
    Compute soft minimum of triangle areas and its gradient.
    soft_min = -1/beta * log(sum(exp(-beta * A_k)))
    grad = sum( p_k * grad_A_k )
    where p_k = exp(-beta * A_k) / sum(exp(-beta * A_j))
    """
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Store areas and indices for gradient calculation
    # To save memory and time, we compute areas and gradients in one pass
    # But vectorizing the gradient of all triangles is complex.
    # We will compute areas first, then weights, then accumulate gradients.
    
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0, np.zeros_like(pts)
    
    # Calculate weights p_k
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
        weights = np.zeros_like(areas)
        if len(areas) > 0:
            weights[np.argmax(z)] = 1.0 # Fallback to hard min if underflow
    else:
        exp_z = np.exp(z - max_z)
        sum_exp = np.sum(exp_z)
        log_sum = max_z + np.log(sum_exp)
        weights = exp_z / sum_exp
        
    soft_min_val = -log_sum / beta
    
    # Calculate gradient
    grad = np.zeros_like(pts)
    
    # We need to iterate through triangles again to apply weights to gradients
    # This is O(N^3) but with small constants.
    idx = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                w = weights[idx]
                if w > 1e-12: # Skip negligible weights
                    # Area A = 0.5 * | (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i) |
                    # Let cross = (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)
                    # A = 0.5 * |cross|
                    # dA/dx = 0.5 * sign(cross) * d(cross)/dx
                    
                    xi, yi = x[i], y[i]
                    xj, yj = x[j], y[j]
                    xk, yk = x[k], y[k]
                    
                    cross = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
                    sign_c = np.sign(cross)
                    if sign_c == 0:
                        sign_c = 1.0 # Avoid zero gradient, arbitrary direction
                    
                    # d(cross)/dx_i = -(yk - yi)
                    # d(cross)/dy_i = (xk - xi)
                    # d(cross)/dx_j = (yk - yi)
                    # d(cross)/dy_j = -(xk - xi)
                    # d(cross)/dx_k = -(yj - yi)
                    # d(cross)/dy_k = (xj - xi)
                    
                    d_cross = np.zeros((n, 2))
                    d_cross[i, 0] = -(yk - yi)
                    d_cross[i, 1] = (xk - xi)
                    d_cross[j, 0] = (yk - yi)
                    d_cross[j, 1] = -(xk - xi)
                    d_cross[k, 0] = -(yj - yi)
                    d_cross[k, 1] = (xj - xi)
                    
                    grad += w * 0.5 * sign_c * d_cross
                idx += 1
                
    return soft_min_val, grad

def _obj_and_grad(pts_flat, beta):
    """
    Objective: HullArea / SoftMinArea
    Gradient: (SoftMin * GradHull - HullArea * GradSoftMin) / SoftMin^2
    """
    pts = pts_flat.reshape(N, 2)
    
    # Check for invalid points
    if np.any(~np.isfinite(pts)):
        return 1e10, np.zeros_like(pts_flat)
    
    # Compute Hull Area and Gradient (Numerical)
    hull_a = _hull_area(pts)
    if hull_a < 1e-10:
        return 1e10, np.zeros_like(pts_flat)
        
    # Numerical gradient for Hull Area
    # Use central difference with small eps
    eps = 1e-6
    grad_hull = np.zeros_like(pts_flat)
    for i in range(N):
        for d in range(2):
            orig = pts_flat[i*2+d]
            pts_p = pts_flat.copy()
            pts_m = pts_flat.copy()
            pts_p[i*2+d] = orig + eps
            pts_m[i*2+d] = orig - eps
            
            hp = _hull_area(pts_p.reshape(N, 2))
            hm = _hull_area(pts_m.reshape(N, 2))
            grad_hull[i*2+d] = (hp - hm) / (2 * eps)
            
    # Compute Soft Min and Gradient (Analytic)
    soft_min_val, grad_soft = _soft_min_and_grad(pts, beta)
    grad_soft_flat = grad_soft.flatten()
    
    if soft_min_val < 1e-12:
        return 1e10, np.zeros_like(pts_flat)
        
    # Objective: H / S
    obj = hull_a / soft_min_val
    
    # Gradient: (S * gradH - H * gradS) / S^2
    grad_obj = (soft_min_val * grad_hull - hull_a * grad_soft_flat) / (soft_min_val * soft_min_val)
    
    return obj, grad_obj

def _init_regular(n):
    """Regular polygon on unit circle."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)

def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 1e-12:
        return pts / np.sqrt(hull_a)
    return pts

def _init_random(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.uniform(-1, 1, size=(n, 2))
    return _normalize_unit_area(pts)

def _init_mixed(n, seed):
    """Mix of outer and inner points."""
    rng = np.random.RandomState(seed)
    outer_n = 10
    inner_n = 4
    angles_out = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    
    angles_in = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    
    pts = np.vstack([outer, inner])
    # Add some noise
    pts += rng.normal(0, 0.05, size=pts.shape)
    return _normalize_unit_area(pts)

def construct():
    start_time = time.time()
    
    # Generate initial points
    inits = [
        _init_regular(N),
        _init_random(N, 1),
        _init_random(N, 2),
        _init_mixed(N, 3),
        _init_mixed(N, 4),
    ]
    
    best_pts = None
    best_score = 0.0
    
    # Optimization settings
    # L-BFGS-B is fast. We can try multiple betas.
    # Start with a moderate beta to ensure smoothness, then refine.
    
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 15:
            break
            
        x0 = init.flatten()
        
        # Stage 1: L-BFGS-B on smooth objective
        # Use a beta that balances smoothness and fidelity.
        # Beta=50 is a good start.
        try:
            res = minimize(
                _obj_and_grad, 
                x0, 
                method='L-BFGS-B',
                jac=True, # Use analytic/numeric gradient provided
                args=(50.0,),
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50}
            )
            if res.x is not None:
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
            
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Stage 2: Polish with higher beta (sharper min)
        if best_pts is not None:
            x1 = best_pts.flatten()
            try:
                res2 = minimize(
                    _obj_and_grad, 
                    x1, 
                    method='L-BFGS-B',
                    jac=True,
                    args=(200.0,),
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
                )
                if res2.x is not None:
                    cand = res2.x.reshape(N, 2)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
            except Exception:
                pass

    # If we still have time, try a few more random perturbations of the best found
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        rng = np.random.RandomState(123)
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 2:
                break
            noise = rng.normal(0, 0.01, size=best_pts.shape)
            perturbed = best_pts + noise
            # Re-normalize
            perturbed = _normalize_unit_area(perturbed)
            
            # Quick polish
            x0 = perturbed.flatten()
            try:
                res = minimize(
                    _obj_and_grad, 
                    x0, 
                    method='L-BFGS-B',
                    jac=True,
                    args=(200.0,),
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
                )
                if res.x is not None:
                    cand = res.x.reshape(N, 2)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
            except Exception:
                pass

    # Fallback
    if best_pts is None:
        best_pts = _init_regular(N)
        
    return best_pts
# EVOLVE-BLOCK-END
