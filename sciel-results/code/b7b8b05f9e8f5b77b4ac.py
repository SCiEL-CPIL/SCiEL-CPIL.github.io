import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_lattice_init(n):
    """Generate an initial hexagonal lattice packing."""
    spacing = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
    points = []
    x = np.arange(-1.2, 1.2, spacing)
    y = np.arange(-1.2, 1.2, spacing)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            offset = 0 if i % 2 == 0 else spacing / 2.0
            yj_offset = yj + offset
            points.append((xi, yj_offset))
            
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    mask = radii <= 1.0 - spacing / 2.0
    points = points[mask]
    
    if len(points) < n:
        spacing *= 0.9
        points = []
        x = np.arange(-1.2, 1.2, spacing)
        y = np.arange(-1.2, 1.2, spacing)
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                offset = 0 if i % 2 == 0 else spacing / 2.0
                yj_offset = yj + offset
                points.append((xi, yj_offset))
        points = np.array(points)
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii <= 1.0 - spacing / 2.0
        points = points[mask]

    if len(points) > n:
        points = points[:n]
        
    return points

def _calculate_radius(points):
    from scipy.spatial.distance import pdist
    dists = pdist(points, metric='euclidean')
    min_dist = np.min(dists)
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(1.0 - radii)
    return min(min_dist / 2.0, min_boundary)

def construct():
    start_time = time.time()
    timeout = 115.0
    
    x0 = _hex_lattice_init(N)
    best_x = x0.copy()
    best_r = _calculate_radius(x0)
    
    # Smooth objective function
    # f(x) = - (1/beta) * log(sum(exp(-beta * d_ij))) + lambda * sum(max(0, r_i - 1)^2)
    # Gradient:
    # d/dx_k [ -1/beta log(sum_j exp(-beta d_kj)) ]
    # = -1/beta * 1/sum_j exp(-beta d_kj) * sum_j exp(-beta d_kj) * (-beta) * d/dx_k d_kj
    # = sum_j exp(-beta d_kj) * d/dx_k d_kj / sum_j exp(-beta d_kj)
    # d/dx_k d_kj = (x_k - x_j) / d_kj
    
    beta = 50.0
    lam = 100.0
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 2)
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        # Avoid division by zero
        dists[dists < 1e-10] = 1e-10
        
        # Mask upper triangle to avoid double counting and self distance
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pair_dists = dists[mask]
        pair_diffs = diff[mask] # Shape (M, 2) where M is number of pairs
        
        # Log-sum-exp for stability
        # log(sum(exp(-beta * d)))
        # max_d = max(pair_dists)
        # log(sum(exp(-beta * (d - max_d)))) - beta * max_d
        # But we want to minimize -1/beta * log_sum_exp
        # So we maximize log_sum_exp
        
        # Let's compute the gradient term by term
        # g_k = sum_j w_j * (x_k - x_j) / d_kj
        # where w_j = exp(-beta * d_kj) / sum_l exp(-beta * d_kl)
        
        # This is tricky to vectorize efficiently for all k at once without O(N^2) memory for weights.
        # But N=300, so 90k pairs.
        
        # Let's compute the sum for each k.
        # exp_terms = np.exp(-beta * dists)
        # exp_terms[~mask] = 0 # Zero out lower triangle and diagonal
        # sum_exp = np.sum(exp_terms, axis=1)
        # 
        # grad_pair = np.zeros_like(x)
        # For each k:
        #   grad_pair[k] = sum_j (exp_terms[k,j] / sum_exp[k]) * (x[k] - x[j]) / dists[k,j]
        
        # This is a bit slow in Python loops.
        # Let's try to vectorize.
        
        exp_terms = np.exp(-beta * dists)
        # Zero out non-pairs
        exp_terms[~mask] = 0.0
        
        sum_exp = np.sum(exp_terms, axis=1)
        # Avoid division by zero
        sum_exp[sum_exp < 1e-10] = 1e-10
        
        # Weights W[k,j] = exp_terms[k,j] / sum_exp[k]
        # W shape (N, N)
        W = exp_terms / sum_exp[:, np.newaxis]
        
        # Grad for pair part:
        # grad_pair[k] = sum_j W[k,j] * (x[k] - x[j]) / dists[k,j]
        # Note: dists[k,j] is in the denominator.
        # Let V[k,j] = W[k,j] / dists[k,j]
        V = W / dists
        V[~mask] = 0.0
        
        # grad_pair[k] = sum_j V[k,j] * (x[k] - x[j])
        # = sum_j V[k,j] * x[k] - sum_j V[k,j] * x[j]
        # = x[k] * sum_j V[k,j] - sum_j V[k,j] * x[j]
        
        sum_V = np.sum(V, axis=1)
        # sum_j V[k,j] * x[j] is a matrix-vector product
        # V shape (N, N), x shape (N, 2)
        # result shape (N, 2)
        VX = V @ x
        
        grad_pair = x * sum_V[:, np.newaxis] - VX
        
        # Boundary part
        radii = np.hypot(x[:, 0], x[:, 1])
        # grad of max(0, r - 1)^2
        # if r > 1: 2 * (r - 1) * (x / r)
        # else: 0
        
        boundary_viol = np.maximum(0, radii - 1.0)
        grad_boundary = np.zeros_like(x)
        mask_b = radii > 1.0
        grad_boundary[mask_b] = 2.0 * boundary_viol[mask_b] / radii[mask_b] * x[mask_b]
        
        grad = -grad_pair + lam * grad_boundary
        
        # Objective value
        # -1/beta * log(sum_exp[k]) summed over k?
        # No, the log-sum-exp is over pairs.
        # The objective was -1/beta * log(sum_{i<j} exp(-beta d_ij))
        # This is a single scalar.
        # sum_exp_pairs = np.sum(exp_terms)
        # obj_pair = -1.0 / beta * np.log(sum_exp_pairs)
        
        # Wait, the gradient derivation above was for sum_k log(sum_j exp...).
        # The standard smooth min is -1/beta log(sum exp(-beta d)).
        # The gradient of this scalar w.r.t x_k is:
        # -1/beta * 1/S * sum_j exp(-beta d_kj) * (-beta) * d(d_kj)/dx_k
        # = 1/S * sum_j exp(-beta d_kj) * d(d_kj)/dx_k
        # This matches the formula above if S = sum_{i<j} exp(-beta d_ij) ?
        # No, S in the denominator for each term should be the total sum.
        # My vectorized code calculated sum_exp per row (sum over j for fixed k).
        # This is incorrect for the global log-sum-exp.
        
        # Let's correct the gradient.
        S_total = np.sum(exp_terms)
        if S_total < 1e-10:
            S_total = 1e-10
            
        # W_global[k,j] = exp_terms[k,j] / S_total
        W_global = exp_terms / S_total
        V_global = W_global / dists
        V_global[~mask] = 0.0
        
        sum_V_global = np.sum(V_global, axis=1)
        VX_global = V_global @ x
        grad_pair_global = x * sum_V_global[:, np.newaxis] - VX_global
        
        grad = -grad_pair_global + lam * grad_boundary
        
        obj_pair = -1.0 / beta * np.log(S_total)
        obj_boundary = lam * np.sum(boundary_viol ** 2)
        
        return obj_pair + obj_boundary, grad

    # Optimize
    x_flat = best_x.flatten()
    
    # We can try to run minimize with the analytical gradient
    try:
        res = minimize(objective_and_grad, x_flat, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
        if res.success or res.fun < best_r * -1: # Check if improved
            new_x = res.x.reshape(-1, 2)
            r_new = _calculate_radius(new_x)
            if r_new > best_r:
                best_x = new_x
                best_r = r_new
    except Exception:
        pass

    # If we have time, do another round with higher beta?
    if time.time() - start_time < timeout - 10:
        beta = 100.0
        def objective_and_grad2(x_flat):
            x = x_flat.reshape(-1, 2)
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            dists[dists < 1e-10] = 1e-10
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            
            exp_terms = np.exp(-beta * dists)
            exp_terms[~mask] = 0.0
            S_total = np.sum(exp_terms)
            if S_total < 1e-10:
                S_total = 1e-10
            W_global = exp_terms / S_total
            V_global = W_global / dists
            V_global[~mask] = 0.0
            sum_V_global = np.sum(V_global, axis=1)
            VX_global = V_global @ x
            grad_pair_global = x * sum_V_global[:, np.newaxis] - VX_global
            
            radii = np.hypot(x[:, 0], x[:, 1])
            boundary_viol = np.maximum(0, radii - 1.0)
            grad_boundary = np.zeros_like(x)
            mask_b = radii > 1.0
            grad_boundary[mask_b] = 2.0 * boundary_viol[mask_b] / radii[mask_b] * x[mask_b]
            
            grad = -grad_pair_global + lam * grad_boundary
            obj_pair = -1.0 / beta * np.log(S_total)
            obj_boundary = lam * np.sum(boundary_viol ** 2)
            return obj_pair + obj_boundary, grad

        try:
            res2 = minimize(objective_and_grad2, best_x.flatten(), method='L-BFGS-B', jac=True, options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8})
            if res2.success:
                new_x2 = res2.x.reshape(-1, 2)
                r_new2 = _calculate_radius(new_x2)
                if r_new2 > best_r:
                    best_x = new_x2
                    best_r = r_new2
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
