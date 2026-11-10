import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Initialization: Hexagonal Lattice
    def gen_hex_lattice():
        # Estimate spacing
        # Area of unit circle is pi. N circles of radius r.
        # Hex packing density ~ 0.907.
        # N * pi * r^2 * 0.907 <= pi  =>  r <= sqrt(0.907/N)
        r_est = np.sqrt(0.907 / N)
        spacing = 2 * r_est * 1.05 # Slightly larger to allow relaxation
        
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.5:
                    pts.append((x, y))
        
        pts = np.array(pts)
        if len(pts) < N:
            # Fallback: random
            pts = np.random.uniform(-0.9, 0.9, (N, 2))
        else:
            # Select N closest to center
            radii = np.linalg.norm(pts, axis=1)
            indices = np.argsort(radii, kind='stable')[:N]
            pts = pts[indices]
            
        return pts

    points = gen_hex_lattice()
    
    # Compute initial score
    def compute_score(pts):
        n = len(pts)
        if n < 2:
            return 0.0
        dist = cdist(pts, pts)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        radii = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    best_score = compute_score(points)
    best_points = points.copy()

    # 2. Optimization using Soft-Min Objective
    # Objective: Maximize soft_min(distances)
    # soft_min(D) = -1/beta * log(sum(exp(-beta * d)))
    # Gradient is weighted sum of gradients of d, weights are softmax.
    
    def soft_min_objective(x, beta, active_mask_cache=None):
        pts = x.reshape(-1, 2)
        n = len(pts)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Boundary distances
        radii = np.linalg.norm(pts, axis=1)
        bdist = 1.0 - radii
        
        # We only care about distances that are close to the minimum
        # to keep the computation fast and stable.
        # Estimate current min
        min_pair = np.min(dist)
        min_bound = np.min(bdist)
        r_min = min(min_pair / 2.0, min_bound)
        
        # Threshold for active constraints
        # Include pairs with dist < 2 * (r_min + epsilon)
        # And boundaries with bdist < r_min + epsilon
        eps = 0.05
        threshold = 2.0 * (r_min + eps)
        
        # Active pairs
        pair_mask = dist < threshold
        np.fill_diagonal(pair_mask, False)
        
        # Active boundaries
        bound_mask = bdist < (r_min + eps)
        
        if not np.any(pair_mask) and not np.any(bound_mask):
            return -r_min, pts, pair_mask, bound_mask, diff, dist, bdist, radii

        # Calculate weights
        # d_values: 0.5 * dist for pairs, bdist for boundaries
        # Note: The score is min(min_pair/2, min_bound).
        # So we treat d_pair = dist/2 and d_bound = bdist.
        
        d_pairs = dist / 2.0
        d_bonds = bdist
        
        # Masked values
        d_pairs_active = d_pairs[pair_mask]
        d_bonds_active = d_bonds[bound_mask]
        
        all_d = np.concatenate([d_pairs_active, d_bonds_active])
        
        if len(all_d) == 0:
             return -r_min, pts, pair_mask, bound_mask, diff, dist, bdist, radii

        # Softmax weights
        # w_k = exp(-beta * d_k) / sum(exp(-beta * d_m))
        # To avoid overflow/underflow, subtract min
        d_min = np.min(all_d)
        exp_terms = np.exp(-beta * (all_d - d_min))
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # Objective value
        soft_min_val = d_min - np.log(sum_exp) / beta
        # We want to MAXIMIZE soft_min_val, so return NEGATIVE for minimizer
        obj_val = -soft_min_val
        
        # Gradient Calculation
        # grad_obj = - sum_k w_k * grad(d_k)
        # Wait, soft_min = d_min - (1/beta) log(sum exp).
        # d(soft_min)/dx = - (1/beta) * d(log(sum exp))/dx
        # d(log(sum exp))/dx = (1/sum exp) * sum (exp * d(-beta d_k)/dx)
        # = (1/sum exp) * sum (exp * -beta * d(d_k)/dx)
        # = -beta * sum (w_k * d(d_k)/dx)
        # So d(soft_min)/dx = - (1/beta) * (-beta) * sum (w_k * d(d_k)/dx)
        # = sum (w_k * d(d_k)/dx)
        
        # We want to minimize obj_val = -soft_min_val
        # So grad(obj_val) = - grad(soft_min_val) = - sum (w_k * d(d_k)/dx)
        
        grad = np.zeros_like(pts)
        
        # Weights for pairs
        w_pairs = np.zeros_like(dist)
        w_pairs[pair_mask] = weights[:len(d_pairs_active)]
        
        # Weights for boundaries
        w_bonds = np.zeros(n)
        w_bonds[bound_mask] = weights[len(d_pairs_active):]
        
        # Gradient of d_pair = dist/2
        # d(dist)/d(c_i) = (c_i - c_j) / dist
        # d(d_pair)/d(c_i) = 0.5 * (c_i - c_j) / dist = (c_i - c_j) / (2 * dist)
        
        # Vectorized gradient for pairs
        # diff[i,j] = c_i - c_j
        # grad_dist[i,j] = diff[i,j] / dist[i,j]
        # We need to sum over j for each i.
        # term_i = sum_j w_pairs[i,j] * (c_i - c_j) / (2 * dist[i,j])
        
        # To avoid division by zero, we masked dist < inf, but dist can be 0 if points coincide.
        # In practice, with random init, they don't coincide.
        # Let's be safe.
        safe_dist = np.where(dist > 1e-10, dist, 1.0)
        
        # grad_pair_contrib[i,j] = w_pairs[i,j] * diff[i,j] / (2 * safe_dist[i,j])
        # Note: diff is (N, N, 2).
        
        grad_pairs = w_pairs[:, :, np.newaxis] * diff / (2.0 * safe_dist[:, :, np.newaxis])
        grad_from_pairs = np.sum(grad_pairs, axis=1)
        
        # Gradient of d_bound = 1 - |c_i|
        # d(|c_i|)/d(c_i) = c_i / |c_i|
        # d(d_bound)/d(c_i) = - c_i / |c_i|
        
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        grad_from_bonds = -w_bonds[:, np.newaxis] * pts / safe_radii[:, np.newaxis]
        
        # Total gradient of soft_min
        grad_soft_min = grad_from_pairs + grad_from_bonds
        
        # Gradient of objective (negative soft_min)
        grad_obj = -grad_soft_min
        
        return obj_val, pts, pair_mask, bound_mask, diff, dist, bdist, radii, grad_obj

    # L-BFGS-B wrapper
    def lfgs_func(x):
        # We need to pass beta. We'll use a fixed beta for now, or adapt it.
        # Let's start with a moderate beta.
        beta = 50.0
        res = soft_min_objective(x, beta)
        if len(res) == 9:
            val, _, _, _, _, _, _, _, grad = res
        else:
            # Fallback if no active constraints (shouldn't happen often)
            val, pts, pair_mask, bound_mask, diff, dist, bdist, radii = res
            grad = np.zeros_like(pts).flatten()
        return val, grad.flatten()

    # Run optimization
    # We will run multiple iterations with increasing beta to sharpen the minimum.
    current_points = best_points.copy()
    
    for beta in [10.0, 20.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline - 20:
            break
            
        # Define objective with fixed beta for this stage
        def obj_func(x, b=beta):
            res = soft_min_objective(x, b)
            if len(res) == 9:
                val, _, _, _, _, _, _, _, grad = res
            else:
                val, pts, pair_mask, bound_mask, diff, dist, bdist, radii = res
                grad = np.zeros_like(pts).flatten()
            return val, grad.flatten()

        try:
            result = minimize(obj_func, current_points.flatten(), method='L-BFGS-B', 
                              jac=True, options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10})
            
            if result.success or result.fun < obj_func(current_points.flatten())[0]:
                new_points = result.x.reshape(-1, 2)
                # Project inside circle
                radii = np.linalg.norm(new_points, axis=1)
                mask_out = radii > 1.0
                if np.any(mask_out):
                    new_points[mask_out] /= radii[mask_out, np.newaxis]
                
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
                    current_points = new_points
        except Exception:
            pass

    # Final Polish with Nelder-Mead on critical points
    if time.time() < deadline - 5:
        n = len(best_points)
        dist = cdist(best_points, best_points)
        np.fill_diagonal(dist, np.inf)
        flat_dist = dist.flatten()
        k_small = min(20, n * (n - 1) // 2)
        smallest_indices = np.argpartition(flat_dist, k_small)[:k_small]
        critical_points = set()
        for idx in smallest_indices:
            i, j = np.unravel_index(idx, (n, n))
            critical_points.add(i)
            critical_points.add(j)
        
        radii = np.linalg.norm(best_points, axis=1)
        boundary_crit = np.where(radii > 0.95)[0]
        for idx in boundary_crit:
            critical_points.add(idx)
        
        if 0 < len(critical_points) < n:
            crit_indices = sorted(critical_points)
            
            def polish_objective(x):
                new_pts = best_points.copy()
                new_pts[crit_indices] = x.reshape(-1, 2)
                return -compute_score(new_pts)
            
            x0_polish = best_points[crit_indices].flatten()
            try:
                result_polish = minimize(polish_objective, x0_polish, method='Nelder-Mead',
                                        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                                  'adaptive': True})
                if result_polish.success or result_polish.fun < polish_objective(x0_polish):
                    new_pts = best_points.copy()
                    new_pts[crit_indices] = result_polish.x.reshape(-1, 2)
                    radii = np.linalg.norm(new_pts, axis=1)
                    mask_out = radii > 1.0
                    if np.any(mask_out):
                        new_pts[mask_out] /= radii[mask_out, np.newaxis]
                    new_score = compute_score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts
            except Exception:
                pass

    return best_points
