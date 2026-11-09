import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def get_score(points):
    """
    Calculate the score: minimum triangle area divided by convex hull area.
    """
    N = len(points)
    if N < 3:
        return 0.0

    # Check for degeneracy
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized triangle area calculation
    # For N=13, M=286 triangles.
    idx = np.indices((N, N, N))
    i, j, k = idx
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    dx_ji = p_j[:, 0] - p_i[:, 0]
    dy_ji = p_j[:, 1] - p_i[:, 1]
    dx_ki = p_k[:, 0] - p_i[:, 0]
    dy_ki = p_k[:, 1] - p_i[:, 1]
    
    cross = dx_ji * dy_ki - dy_ji * dx_ki
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    N = 13
    best_score = 0.0
    best_points = None
    
    # 1. Baseline: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = get_score(points_reg)
    best_points = points_reg.copy()
    
    # Pre-compute triangle indices for speed
    idx = np.indices((N, N, N))
    i_idx, j_idx, k_idx = idx
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    num_tris = len(i_idx)
    
    def soft_min_score_and_grad(x_flat):
        """
        Objective: - (soft_min(triangle_areas) / hull_area)
        Gradient: analytic gradient of soft-min w.r.t coordinates.
        
        Soft min: m = -1/beta * log(sum(exp(-beta * a_t)))
        grad m / grad a_t = exp(-beta * a_t) / sum(exp(-beta * a_s))
        
        a_t = 0.5 * |cross|
        grad a_t / grad p_v = 0.5 * sign(cross) * d(cross)/d(p_v)
        """
        pts = x_flat.reshape(N, 2)
        
        # Calculate areas
        p_i = pts[i_idx]
        p_j = pts[j_idx]
        p_k = pts[k_idx]
        
        dx_ji = p_j[:, 0] - p_i[:, 0]
        dy_ji = p_j[:, 1] - p_i[:, 1]
        dx_ki = p_k[:, 0] - p_i[:, 0]
        dy_ki = p_k[:, 1] - p_i[:, 1]
        
        cross = dx_ji * dy_ki - dy_ji * dx_ki
        areas = 0.5 * np.abs(cross)
        
        # Calculate hull area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 1e6, np.zeros_like(x_flat)
        except Exception:
            return 1e6, np.zeros_like(x_flat)
            
        # Soft min parameters
        # Beta needs to be large enough to approximate min, but not so large that exp overflows/underflows numerically.
        # Areas are typically O(0.01) to O(0.1). 
        # Let's normalize areas for the softmax to be stable.
        min_area = np.min(areas)
        
        # To avoid overflow in exp(-beta * area), we can subtract min_area.
        # exp(-beta * (a - min_a)) = exp(-beta * a) * exp(beta * min_a)
        # The factor exp(beta * min_a) cancels in the ratio for weights, but scales the soft_min value.
        # soft_min = min_a - (1/beta) * log(sum(exp(-beta * (a - min_a))))
        
        beta = 200.0
        
        shifted = areas - min_area
        exp_terms = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            return 1e6, np.zeros_like(x_flat)
            
        soft_min_val = min_area - (1.0 / beta) * np.log(sum_exp)
        
        # Score is soft_min / hull_area
        score = soft_min_val / hull_area
        
        # Gradient calculation
        # d(score) / d(p_v) = (1/hull_area) * d(soft_min) / d(p_v)
        # d(soft_min) / d(p_v) = sum_t [ w_t * d(a_t) / d(p_v) ]
        # where w_t = exp(-beta * shifted_t) / sum_exp
        
        weights = exp_terms / sum_exp
        
        # d(a_t) / d(p_v)
        # a_t = 0.5 * |cross_t|
        # cross_t = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
        # d(cross)/d(x_i) = -(y_k - y_i) = y_i - y_k
        # d(cross)/d(y_i) = (x_k - x_i) = x_k - x_i
        # d(cross)/d(x_j) = (y_k - y_i)
        # d(cross)/d(y_j) = -(x_k - x_i) = x_i - x_k
        # d(cross)/d(x_k) = -(y_j - y_i) = y_i - y_j
        # d(cross)/d(y_k) = (x_j - x_i)
        
        # sign(cross) is needed because of the abs()
        signs = np.sign(cross)
        # If cross is 0, sign is 0, which is fine (subgradient 0)
        
        # Gradient of cross w.r.t coordinates for each triangle
        # We need to accumulate contributions to each point's x and y
        
        grad_flat = np.zeros_like(x_flat)
        
        # Vectorized accumulation is tricky with variable indices.
        # For N=13 and 286 triangles, a loop is fast enough (286 iterations).
        # But let's try to vectorize using np.add.at or similar if possible.
        # Actually, 286 is small. A Python loop might be slower than vectorized ops if we can do it.
        # Let's use np.add.at for efficiency.
        
        # d_cross_dx_i = -(y_k - y_i) = y_i - y_k
        # d_cross_dy_i = (x_k - x_i)
        # d_cross_dx_j = (y_k - y_i)
        # d_cross_dy_j = -(x_k - x_i) = x_i - x_k
        # d_cross_dx_k = -(y_j - y_i) = y_i - y_j
        # d_cross_dy_k = (x_j - x_i)
        
        # Factor 0.5 * sign(cross) * weight_t
        coeff = 0.5 * signs * weights
        
        # Accumulate for point i
        # grad_x_i += coeff * (y_i - y_k)
        # grad_y_i += coeff * (x_k - x_i)
        
        # We can precompute the terms
        # term_i_x = coeff * (y_i - y_k)
        # term_i_y = coeff * (x_k - x_i)
        
        # Wait, y_i depends on which point is i. 
        # p_i is an array of shape (num_tris, 2).
        
        dy_i_k = p_i[:, 1] - p_k[:, 1]
        dx_k_i = p_k[:, 0] - p_i[:, 0]
        
        dx_j_i = p_j[:, 0] - p_i[:, 0]
        dy_k_i = p_k[:, 1] - p_i[:, 1]
        dy_j_i = p_j[:, 1] - p_i[:, 1]
        dx_i_k = p_i[:, 0] - p_k[:, 0]
        dy_i_j = p_i[:, 1] - p_j[:, 1]
        dx_j_i2 = p_j[:, 0] - p_i[:, 0] # same as dx_j_i
        
        # For point i:
        contrib_i_x = coeff * dy_i_k
        contrib_i_y = coeff * dx_k_i
        
        # For point j:
        contrib_j_x = coeff * dy_k_i
        contrib_j_y = coeff * dx_i_k
        
        # For point k:
        contrib_k_x = coeff * dy_i_j
        contrib_k_y = coeff * dx_j_i2
        
        # np.add.at accumulates
        np.add.at(grad_flat, i_idx * 2, contrib_i_x)
        np.add.at(grad_flat, i_idx * 2 + 1, contrib_i_y)
        
        np.add.at(grad_flat, j_idx * 2, contrib_j_x)
        np.add.at(grad_flat, j_idx * 2 + 1, contrib_j_y)
        
        np.add.at(grad_flat, k_idx * 2, contrib_k_x)
        np.add.at(grad_flat, k_idx * 2 + 1, contrib_k_y)
        
        # Divide by hull_area
        grad_flat /= hull_area
        
        # We want to maximize score, so minimize -score
        # Gradient of -score is -grad_flat
        return -score, -grad_flat

    # 2. Optimization using Soft-Min
    # Start from regular polygon and a few perturbed versions
    
    initial_configs = []
    # Regular
    initial_configs.append(points_reg.flatten().copy())
    
    # Perturbed regular
    for _ in range(3):
        noise = np.random.normal(0, 0.1, N * 2)
        initial_configs.append(points_reg.flatten() + noise)
        
    # Random blob (centered)
    for _ in range(2):
        pts_rand = np.random.uniform(-1, 1, (N, 2))
        # Center it
        pts_rand -= np.mean(pts_rand, axis=0)
        # Scale to roughly unit radius
        max_r = np.max(np.linalg.norm(pts_rand, axis=1))
        if max_r > 1e-6:
            pts_rand /= max_r
        initial_configs.append(pts_rand.flatten())
        
    for cfg in initial_configs:
        if time.time() > deadline - 20:
            break
            
        try:
            # Use L-BFGS-B with analytic gradient
            # Bounds: keep points within a reasonable box, say [-2, 2]
            bounds = [(-2.0, 2.0)] * (N * 2)
            
            res = minimize(soft_min_score_and_grad, cfg, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 20})
            
            current_points = res.x.reshape(N, 2)
            current_score = get_score(current_points)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                
        except Exception:
            pass

    # 3. Final Refinement with Nelder-Mead on the exact min (non-smooth)
    # The soft-min might settle slightly above the true min. 
    # Nelder-Mead on the exact function can polish the result.
    
    if best_points is not None:
        def neg_score_exact(x):
            pts = x.reshape(N, 2)
            s = get_score(pts)
            return -s if s > 0 else 1.0
            
        x0 = best_points.flatten()
        
        # Run a few quick Nelder-Mead restarts from the best soft-min result
        for i in range(5):
            if time.time() > deadline - 5:
                break
            try:
                # Small noise to escape any local flat spots
                noise = np.random.normal(0, 0.005, N * 2)
                x_start = x0 + noise
                
                res = minimize(neg_score_exact, x_start, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = res.x.reshape(N, 2).copy()
                    x0 = best_points.flatten()
            except Exception:
                pass

    return best_points
