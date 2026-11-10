import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s margin

    # Precompute indices for pairs
    # We need gradients of distances wrt points
    # d_k is distance between point i and j
    
    def get_pair_indices(n):
        i, j = np.triu_indices(n, k=1)
        return i, j

    pair_i, pair_j = get_pair_indices(n)
    n_pairs = len(pair_i)

    def compute_dists_grads(pts):
        """
        pts: (n, d)
        Returns:
        dists: (n_pairs,)
        grads: (n_pairs, n, d) - gradient of each distance wrt each point
        """
        # Vectorized distance calculation
        # diff[i, j] = pts[i] - pts[j]
        # We only need upper triangle
        pi = pts[pair_i]
        pj = pts[pair_j]
        diffs = pi - pj # (n_pairs, d)
        dists = np.linalg.norm(diffs, axis=1) # (n_pairs,)
        
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-10, 1e-10, dists)
        
        # grads[k, i, :] = (pi - pj) / dist_k
        # grads[k, j, :] = (pj - pi) / dist_k
        # All other points have 0 gradient
        
        # We can construct the full gradient tensor or just compute the weighted sum directly in the objective gradient.
        # Let's return dists and diffs to save memory/time in the outer loop
        return dists, diffs, dists_safe

    def smooth_objective_and_grad(x, tau):
        """
        x: (n*d,)
        tau: temperature for soft min/max
        Returns: value, gradient
        """
        if time.time() > deadline:
            return 1e10, np.zeros_like(x)
            
        pts = x.reshape((n, d))
        # Center points to remove translation invariance (optional, but helps conditioning)
        # Actually, scaling invariance is also an issue. 
        # The ratio d_min/d_max is invariant to scaling and translation.
        # To help the optimizer, we can fix the scale, e.g., d_max = 1.
        # But d_max is variable. 
        # Let's just center.
        pts = pts - np.mean(pts, axis=0)
        
        dists, diffs, dists_safe = compute_dists_grads(pts)
        
        if np.any(dists < 1e-10):
            return 1e10, np.zeros_like(x)

        # Soft Min
        # S_min = -tau * log(sum(exp(-dists/tau)))
        # For numerical stability, subtract max(-dists/tau) which is min(dists)/tau
        neg_dists_tau = -dists / tau
        max_neg_dists_tau = np.max(neg_dists_tau)
        exp_neg = np.exp(neg_dists_tau - max_neg_dists_tau)
        sum_exp_neg = np.sum(exp_neg)
        S_min = -tau * (np.log(sum_exp_neg) + max_neg_dists_tau)
        
        # Weights for S_min gradient: w_min_k = exp(-d_k/tau) / sum(exp(-d_k/tau))
        # w_min_k = exp_neg_k / sum_exp_neg
        w_min = exp_neg / sum_exp_neg

        # Soft Max
        # S_max = tau * log(sum(exp(dists/tau)))
        pos_dists_tau = dists / tau
        max_pos_dists_tau = np.max(pos_dists_tau)
        exp_pos = np.exp(pos_dists_tau - max_pos_dists_tau)
        sum_exp_pos = np.sum(exp_pos)
        S_max = tau * (np.log(sum_exp_pos) + max_pos_dists_tau)
        
        if S_max < 1e-10:
            return 1e10, np.zeros_like(x)

        w_max = exp_pos / sum_exp_pos
        
        val = (S_min / S_max) ** 2
        
        # Gradient
        # grad_val = 2 * (S_min / S_max^2) * grad_S_min - 2 * (S_min^2 / S_max^3) * grad_S_max
        
        # grad_S_min = sum_k w_min_k * grad_d_k
        # grad_S_max = sum_k w_max_k * grad_d_k
        
        # grad_d_k wrt pts[i] = diffs[k] / dists_safe[k]
        # grad_d_k wrt pts[j] = -diffs[k] / dists_safe[k]
        
        # We need to accumulate these into a (n, d) gradient
        grad_pts = np.zeros_like(pts)
        
        # Unit vectors
        unit_vecs = diffs / dists_safe[:, np.newaxis] # (n_pairs, d)
        
        # Contribution to min
        # w_min is (n_pairs,)
        # unit_vecs is (n_pairs, d)
        # We need to scatter these to points i and j
        
        # grad_S_min = sum_k w_min_k * (unit_vec_k at i, -unit_vec_k at j)
        # This is equivalent to:
        # grad_S_min[i] += sum_{k: i_k == i} w_min_k * unit_vec_k
        # grad_S_min[j] -= sum_{k: j_k == j} w_min_k * unit_vec_k
        
        # Using np.add.at for efficient scattering
        np.add.at(grad_pts, pair_i, w_min[:, np.newaxis] * unit_vecs)
        np.add.at(grad_pts, pair_j, -w_min[:, np.newaxis] * unit_vecs)
        
        grad_S_min = grad_pts
        
        # Reset grad_pts for max
        grad_pts = np.zeros_like(pts)
        
        np.add.at(grad_pts, pair_i, w_max[:, np.newaxis] * unit_vecs)
        np.add.at(grad_pts, pair_j, -w_max[:, np.newaxis] * unit_vecs)
        
        grad_S_max = grad_pts
        
        coeff_min = 2 * (S_min / (S_max ** 2))
        coeff_max = -2 * (S_min ** 2 / (S_max ** 3))
        
        grad_val_pts = coeff_min * grad_S_min + coeff_max * grad_S_max
        
        # Chain rule: x -> pts
        # pts = x.reshape(n,d) - mean
        # The gradient wrt x is the same as wrt pts, because mean is a linear operation 
        # and the objective is translation invariant? 
        # Wait. If we center the points inside the objective, the gradient wrt the original x 
        # is slightly different due to the centering step.
        # Let y = x - mean(x). f(y).
        # df/dx = df/dy * dy/dx.
        # y_i = x_i - (1/n) sum x_j.
        # dy_i/dx_k = delta_ik - 1/n.
        # This is a projection. Since the objective is translation invariant, 
        # the gradient in the direction of translation is 0.
        # So we can just return grad_val_pts flattened.
        
        return val, grad_val_pts.flatten()

    def exact_score(pts):
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    best_score = -np.inf
    best_points = None
    
    rng = np.random.RandomState(42)
    
    # Initial guesses
    # 1. Random sphere
    # 2. Icosahedron + 2
    phi = (1 + np.sqrt(5)) / 2
    ico12 = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico12 = ico12 / np.linalg.norm(ico12, axis=1, keepdims=True)
    
    inits = []
    # Random
    for _ in range(10):
        inits.append(rng.randn(n, d))
    # Icosahedron based
    for _ in range(5):
        extra = rng.randn(2, 3) * 0.5
        inits.append(np.vstack([ico12, extra]))
    # Fibonacci
    def fibonacci_sphere(n_pts):
        indices = np.arange(0, n_pts, dtype=float) + 0.5
        phi_f = np.arccos(1 - 2 * indices / n_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi_f)
        y = np.sin(theta) * np.sin(phi_f)
        z = np.cos(phi_f)
        return np.column_stack([x, y, z])
    fib14 = fibonacci_sphere(14)
    inits.append(fib14)
    for _ in range(5):
        inits.append(fib14 + rng.randn(n, d) * 0.1)

    for init in inits:
        if time.time() > deadline - 30:
            break
            
        x0 = init.flatten()
        
        # Annealing schedule for tau
        # Start with large tau (smooth), end with small tau (sharp)
        taus = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        
        current_x = x0
        for tau in taus:
            if time.time() > deadline - 10:
                break
            
            try:
                res = minimize(
                    smooth_objective_and_grad,
                    current_x,
                    args=(tau,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < 1e9:
                    current_x = res.x
            except Exception:
                pass
        
        # Final polish with exact objective using Nelder-Mead
        # This corrects for the bias of the soft approximation
        try:
            res_nm = minimize(
                lambda x: -exact_score(x.reshape(n, d)),
                current_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts = res_nm.x.reshape(n, d)
            s = exact_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
        except Exception:
            pts = current_x.reshape(n, d)
            s = exact_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts

    # If we have time, do more random restarts
    if best_points is not None and time.time() < deadline - 60:
        for _ in range(20):
            if time.time() > deadline - 30:
                break
            x0 = rng.randn(n, d).flatten()
            taus = [0.5, 0.1, 0.01]
            current_x = x0
            for tau in taus:
                if time.time() > deadline - 10:
                    break
                try:
                    res = minimize(
                        smooth_objective_and_grad,
                        current_x,
                        args=(tau,),
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
                    )
                    if res.fun < 1e9:
                        current_x = res.x
                except Exception:
                    pass
            
            try:
                res_nm = minimize(
                    lambda x: -exact_score(x.reshape(n, d)),
                    current_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                pts = res_nm.x.reshape(n, d)
                s = exact_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
            except Exception:
                pass

    if best_points is None:
        best_points = inits[0]
        
    # Normalize output
    best_points = best_points - np.mean(best_points, axis=0)
    norm_max = np.max(np.linalg.norm(best_points, axis=1))
    if norm_max > 0:
        best_points = best_points / norm_max * 100

    return best_points
