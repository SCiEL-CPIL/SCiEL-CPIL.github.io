# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def heilbronn_convex13() -> np.ndarray:
    n = 13
    # Precompute indices for all 286 triangles
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_i = i_idx[mask]
    tri_j = j_idx[mask]
    tri_k = k_idx[mask]
    num_tris = len(tri_i)
    
    # Time limit guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    best_score = -np.inf
    best_points = None
    
    # Initial configurations
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    base_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # List of initial points to try
    initializations = [base_poly]
    
    # Add some random perturbations of the base polygon
    rng = np.random.default_rng(42)
    for _ in range(5):
        noise = rng.normal(0, 0.1, (n, 2))
        initializations.append(base_poly + noise)
        
    # Add a few fully random points
    for _ in range(2):
        initializations.append(rng.random((n, 2)))

    beta_schedule = [10, 50, 200, 1000]
    
    for init_pts in initializations:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        current_pts = init_pts.copy()
        
        for beta in beta_schedule:
            if time.time() - start_time > time_limit - safety_margin:
                break
                
            def obj_and_grad(p_flat):
                pts = p_flat.reshape(n, 2)
                
                # 1. Compute triangle areas
                # Vectorized calculation
                p_i = pts[tri_i]
                p_j = pts[tri_j]
                p_k = pts[tri_k]
                
                # Area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
                areas = 0.5 * np.abs(
                    p_i[:, 0] * (p_j[:, 1] - p_k[:, 1]) +
                    p_j[:, 0] * (p_k[:, 1] - p_i[:, 1]) +
                    p_k[:, 0] * (p_i[:, 1] - p_j[:, 1])
                )
                
                # Soft-min
                # S = -1/beta * ln(sum(exp(-beta * areas)))
                # To avoid overflow/underflow, subtract max
                max_area = np.max(areas)
                # Actually, for soft-min of positive numbers, we want to emphasize small areas.
                # exp(-beta * area) is large for small area.
                # So we are fine.
                exp_terms = np.exp(-beta * areas)
                sum_exp = np.sum(exp_terms)
                
                if sum_exp == 0:
                    return -1e10, np.zeros_like(p_flat)
                    
                soft_min_val = -1.0 / beta * np.log(sum_exp)
                
                # Gradient of soft_min_val wrt pts
                # dS/dA_k = (exp(-beta A_k) / sum_exp)
                weights = exp_terms / sum_exp
                
                # Gradient of each area A_k wrt pts
                # A_k = 0.5 * |...|
                # Assuming non-collinear, sign is constant locally.
                # Let's compute the signed area first to get gradients, then take abs?
                # No, the absolute value makes it non-differentiable at 0.
                # But areas should be > 0.
                # Signed area S_k = 0.5 * (x_i(y_j - y_k) + ...)
                # dS_k/dx_i = 0.5 * (y_j - y_k)
                # dS_k/dy_i = 0.5 * (x_k - x_j)
                
                # We need d(0.5 * |S_k|)/d pts = 0.5 * sign(S_k) * dS_k/d pts
                # Let's compute signed areas
                signed_areas = 0.5 * (
                    p_i[:, 0] * (p_j[:, 1] - p_k[:, 1]) +
                    p_j[:, 0] * (p_k[:, 1] - p_i[:, 1]) +
                    p_k[:, 0] * (p_i[:, 1] - p_j[:, 1])
                )
                signs = np.sign(signed_areas)
                # If area is 0, sign is 0, gradient is 0 (which is fine, we want to move away)
                
                grad_soft_min = np.zeros_like(pts)
                
                # For each triangle k, add weight_k * 0.5 * sign_k * grad_S_k
                # grad_S_k wrt x_i: 0.5 * (y_j - y_k)
                # grad_S_k wrt y_i: 0.5 * (x_k - x_j)
                
                # We can accumulate using np.add.at
                # dA_k/dx_i
                d_dx_i = 0.5 * (p_j[:, 1] - p_k[:, 1])
                d_dy_i = 0.5 * (p_k[:, 0] - p_j[:, 0])
                
                d_dx_j = 0.5 * (p_k[:, 1] - p_i[:, 1])
                d_dy_j = 0.5 * (p_i[:, 0] - p_k[:, 0])
                
                d_dx_k = 0.5 * (p_i[:, 1] - p_j[:, 1])
                d_dy_k = 0.5 * (p_j[:, 0] - p_i[:, 0])
                
                # Weighted contributions
                w = weights * signs * 0.5 # The 0.5 is from d(0.5*|S|)/dS = 0.5*sign(S)
                
                np.add.at(grad_soft_min, (tri_i, 0), w * d_dx_i)
                np.add.at(grad_soft_min, (tri_i, 1), w * d_dy_i)
                np.add.at(grad_soft_min, (tri_j, 0), w * d_dx_j)
                np.add.at(grad_soft_min, (tri_j, 1), w * d_dy_j)
                np.add.at(grad_soft_min, (tri_k, 0), w * d_dx_k)
                np.add.at(grad_soft_min, (tri_k, 1), w * d_dy_k)
                
                # 2. Compute Hull Area
                try:
                    hull = ConvexHull(pts)
                    hull_area = hull.volume
                    hull_idx = hull.vertices
                except Exception:
                    # Invalid hull, return penalty
                    return -1e10, np.zeros_like(p_flat)
                    
                if hull_area < 1e-10:
                    return -1e10, np.zeros_like(p_flat)
                    
                log_hull_area = np.log(hull_area)
                
                # Gradient of log_hull_area
                # A = 0.5 * sum (x_m y_{m+1} - x_{m+1} y_m)
                # dA/dx_m = 0.5 * (y_{m-1} - y_{m+1})
                # dA/dy_m = 0.5 * (x_{m+1} - x_{m-1})
                
                grad_log_hull = np.zeros_like(pts)
                
                # Hull vertices are in CCW order
                # We need to map the hull indices to the original point indices
                # hull_idx contains indices into pts
                
                # Let's compute the gradient for each hull vertex
                for m in range(len(hull_idx)):
                    curr = hull_idx[m]
                    prev = hull_idx[(m - 1) % len(hull_idx)]
                    next_ = hull_idx[(m + 1) % len(hull_idx)]
                    
                    dA_dx = 0.5 * (pts[prev, 1] - pts[next_, 1])
                    dA_dy = 0.5 * (pts[next_, 0] - pts[prev, 0])
                    
                    grad_log_hull[curr, 0] += dA_dx / hull_area
                    grad_log_hull[curr, 1] += dA_dy / hull_area
                    
                # Objective: Maximize soft_min - log_hull_area
                # Minimize: -(soft_min - log_hull_area) = log_hull_area - soft_min
                val = log_hull_area - soft_min_val
                grad = grad_log_hull - grad_soft_min
                
                return val, grad.flatten()

            # Run L-BFGS-B
            try:
                res = minimize(
                    obj_and_grad,
                    current_pts.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                current_pts = res.x.reshape(n, 2)
            except Exception:
                pass

            # Update best
            # Calculate actual score
            try:
                hull = ConvexHull(current_pts)
                hull_area = hull.volume
                # Compute min area
                p_i = current_pts[tri_i]
                p_j = current_pts[tri_j]
                p_k = current_pts[tri_k]
                areas = 0.5 * np.abs(
                    p_i[:, 0] * (p_j[:, 1] - p_k[:, 1]) +
                    p_j[:, 0] * (p_k[:, 1] - p_i[:, 1]) +
                    p_k[:, 0] * (p_i[:, 1] - p_j[:, 1])
                )
                min_area = np.min(areas)
                score = min_area / hull_area
            except Exception:
                score = -np.inf
                
            if score > best_score:
                best_score = score
                best_points = current_pts.copy()
                
    if best_points is None:
        # Fallback
        return base_poly
        
    return best_points

# EVOLVE-BLOCK-END
