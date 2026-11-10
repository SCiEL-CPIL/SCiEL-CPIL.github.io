<<<<<<< SEARCH
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
=======
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
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
    
    def compute_score(pts):
        """Compute the actual score: min_triangle_area / hull_area"""
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return -np.inf
            p_i = pts[tri_i]
            p_j = pts[tri_j]
            p_k = pts[tri_k]
            areas = 0.5 * np.abs(
                p_i[:, 0] * (p_j[:, 1] - p_k[:, 1]) +
                p_j[:, 0] * (p_k[:, 1] - p_i[:, 1]) +
                p_k[:, 0] * (p_i[:, 1] - p_j[:, 1])
            )
            min_area = np.min(areas)
            if min_area < 1e-10:
                return -np.inf
            return min_area / hull_area
        except Exception:
            return -np.inf
    
    def objective_for_de(p_flat):
        """Objective for differential evolution (minimize negative score)"""
        pts = p_flat.reshape(n, 2)
        score = compute_score(pts)
        if score <= 0 or np.isinf(score) or np.isnan(score):
            return 1e10
        return -score
    
    def local_refine(pts, beta_schedule=None, max_iter=200):
        """Refine points using L-BFGS-B with soft-min objective"""
        if beta_schedule is None:
            beta_schedule = [20, 100, 500]
        
        current_pts = pts.copy()
        
        for beta in beta_schedule:
            if time.time() - start_time > time_limit - safety_margin:
                break
                
            def obj_and_grad(p_flat):
                pts = p_flat.reshape(n, 2)
                
                p_i = pts[tri_i]
                p_j = pts[tri_j]
                p_k = pts[tri_k]
                
                signed_areas = 0.5 * (
                    p_i[:, 0] * (p_j[:, 1] - p_k[:, 1]) +
                    p_j[:, 0] * (p_k[:, 1] - p_i[:, 1]) +
                    p_k[:, 0] * (p_i[:, 1] - p_j[:, 1])
                )
                areas = np.abs(signed_areas)
                
                # Soft-min: -1/beta * ln(sum(exp(-beta * areas)))
                max_area = np.max(areas)
                exp_terms = np.exp(-beta * (areas - max_area))
                sum_exp = np.sum(exp_terms)
                
                if sum_exp == 0 or not np.isfinite(sum_exp):
                    return 1e10, np.zeros_like(p_flat)
                    
                soft_min_val = max_area - 1.0 / beta * np.log(sum_exp)
                
                weights = exp_terms / sum_exp
                
                signs = np.sign(signed_areas)
                signs[signs == 0] = 1.0  # Avoid zero gradient
                
                grad_soft_min = np.zeros_like(pts)
                
                d_dx_i = 0.5 * (p_j[:, 1] - p_k[:, 1])
                d_dy_i = 0.5 * (p_k[:, 0] - p_j[:, 0])
                d_dx_j = 0.5 * (p_k[:, 1] - p_i[:, 1])
                d_dy_j = 0.5 * (p_i[:, 0] - p_k[:, 0])
                d_dx_k = 0.5 * (p_i[:, 1] - p_j[:, 1])
                d_dy_k = 0.5 * (p_j[:, 0] - p_i[:, 0])
                
                w = weights * signs * 0.5
                
                np.add.at(grad_soft_min, (tri_i, 0), w * d_dx_i)
                np.add.at(grad_soft_min, (tri_i, 1), w * d_dy_i)
                np.add.at(grad_soft_min, (tri_j, 0), w * d_dx_j)
                np.add.at(grad_soft_min, (tri_j, 1), w * d_dy_j)
                np.add.at(grad_soft_min, (tri_k, 0), w * d_dx_k)
                np.add.at(grad_soft_min, (tri_k, 1), w * d_dy_k)
                
                try:
                    hull = ConvexHull(pts)
                    hull_area = hull.volume
                    hull_idx = hull.vertices
                except Exception:
                    return 1e10, np.zeros_like(p_flat)
                    
                if hull_area < 1e-10:
                    return 1e10, np.zeros_like(p_flat)
                    
                log_hull_area = np.log(hull_area)
                grad_log_hull = np.zeros_like(pts)
                
                for m in range(len(hull_idx)):
                    curr = hull_idx[m]
                    prev = hull_idx[(m - 1) % len(hull_idx)]
                    next_ = hull_idx[(m + 1) % len(hull_idx)]
                    
                    dA_dx = 0.5 * (pts[prev, 1] - pts[next_, 1])
                    dA_dy = 0.5 * (pts[next_, 0] - pts[prev, 0])
                    
                    grad_log_hull[curr, 0] += dA_dx / hull_area
                    grad_log_hull[curr, 1] += dA_dy / hull_area
                    
                val = log_hull_area - soft_min_val
                grad = grad_log_hull - grad_soft_min
                
                if not np.all(np.isfinite(grad)):
                    return 1e10, np.zeros_like(p_flat)
                
                return val, grad.flatten()

            try:
                res = minimize(
                    obj_and_grad,
                    current_pts.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                if res.success or np.isfinite(res.fun):
                    current_pts = res.x.reshape(n, 2)
            except Exception:
                pass
        
        return current_pts
    
    # Strategy 1: Use known good initial configuration
    # Based on literature: for n=13, optimal ratio is around 0.08-0.10
    # Start with a perturbed regular polygon and refine
    
    rng = np.random.default_rng(42)
    
    # Generate multiple diverse initializations
    initializations = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    base_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(base_poly)
    
    # 2. Regular polygon with slight radial perturbation
    for trial in range(3):
        radii = 1.0 + rng.normal(0, 0.15, n)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(pts)
    
    # 3. Random points in unit disk
    for trial in range(3):
        r = np.sqrt(rng.random(n))
        theta = 2 * np.pi * rng.random(n)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        initializations.append(pts)
    
    # 4. Points with some on boundary, some inside
    for trial in range(2):
        boundary_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + rng.uniform(0, 0.3, 8)
        boundary_pts = np.column_stack([np.cos(boundary_angles), np.sin(boundary_angles)])
        interior_r = np.sqrt(rng.random(5)) * 0.5
        interior_theta = 2 * np.pi * rng.random(5)
        interior_pts = np.column_stack([interior_r * np.cos(interior_theta), interior_r * np.sin(interior_theta)])
        pts = np.vstack([boundary_pts, interior_pts])
        initializations.append(pts)
    
    # First pass: local refinement from each initialization
    for init_pts in initializations:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        refined = local_refine(init_pts, beta_schedule=[50, 200], max_iter=100)
        score = compute_score(refined)
        
        if score > best_score:
            best_score = score
            best_points = refined.copy()
    
    # Second pass: differential evolution for global search
    if time.time() - start_time < time_limit - safety_margin:
        # Bounds: points in [-1.5, 1.5]^2
        bounds = [(-1.5, 1.5)] * (n * 2)
        
        de_result = differential_evolution(
            objective_for_de,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            init='latinhypercube',
            updating='immediate'
        )
        
        if de_result.success and np.isfinite(de_result.fun):
            de_pts = de_result.x.reshape(n, 2)
            de_score = compute_score(de_pts)
            
            if de_score > best_score:
                best_score = de_score
                best_points = de_pts.copy()
            
            # Refine DE result
            if time.time() - start_time < time_limit - safety_margin:
                refined_de = local_refine(de_pts, beta_schedule=[50, 200, 500], max_iter=150)
                refined_score = compute_score(refined_de)
                
                if refined_score > best_score:
                    best_score = refined_score
                    best_points = refined_de.copy()
    
    # Third pass: if we have time, do more DE with different parameters
    if time.time() - start_time < time_limit - 60:
        bounds = [(-1.5, 1.5)] * (n * 2)
        
        de_result2 = differential_evolution(
            objective_for_de,
            bounds=bounds,
            seed=123,
            maxiter=30,
            popsize=20,
            tol=1e-5,
            mutation=(0.3, 0.8),
            recombination=0.5,
            polish=False,
            init='sohal',
            updating='immediate'
        )
        
        if de_result2.success and np.isfinite(de_result2.fun):
            de_pts2 = de_result2.x.reshape(n, 2)
            de_score2 = compute_score(de_pts2)
            
            if de_score2 > best_score:
                best_score = de_score2
                best_points = de_pts2.copy()
            
            if time.time() - start_time < time_limit - 30:
                refined_de2 = local_refine(de_pts2, beta_schedule=[50, 200, 500], max_iter=100)
                refined_score2 = compute_score(refined_de2)
                
                if refined_score2 > best_score:
                    best_score = refined_score2
                    best_points = refined_de2.copy()
    
    # Final refinement of best point
    if best_points is not None and time.time() - start_time < time_limit - 15:
        final_refined = local_refine(best_points, beta_schedule=[100, 300, 1000], max_iter=200)
        final_score = compute_score(final_refined)
        
        if final_score > best_score:
            best_score = final_score
            best_points = final_refined.copy()
    
    if best_points is None:
        return base_poly
        
    return best_points

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
