# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for a discrete measure.
    support: 1D array of shape (n,)
    weights: 1D array of shape (n,)
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Normalize weights just in case
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0
    w = w / total_w
    
    # Compute pairwise sums and weights for Z = X1 + X2
    s_sums = s[:, None] + s[None, :]
    p_prods = w[:, None] * w[None, :]
    
    s_sums_flat = s_sums.flatten()
    p_prods_flat = p_prods.flatten()
    
    # Sort pair sums
    sort_idx = np.argsort(s_sums_flat, kind='stable')
    s_sorted = s_sums_flat[sort_idx]
    p_sorted = p_prods_flat[sort_idx]
    p_prefix = np.cumsum(p_sorted)
    
    # Thresholds for X3, X4
    # C[k, l] = P[X1 + X2 < 2*s_l - s_k]
    s_k_grid = s[:, None]  # (n, 1)
    s_l_grid = s[None, :]  # (1, n)
    thresholds = 2.0 * s_l_grid - s_k_grid  # (n, n)
    
    # Searchsorted on sorted s_sums
    idxs = np.searchsorted(s_sorted, thresholds, side='left')
    
    prob_z_less = np.zeros_like(idxs, dtype=np.float64)
    mask = idxs > 0
    if np.any(mask):
        prob_z_less[mask] = p_prefix[np.maximum(idxs[mask] - 1, 0)]
        
    p_k_grid = w[:, None]  # (n, 1)
    p_l_grid = w[None, :]  # (1, n)
    
    total_prob = np.sum(p_k_grid * p_l_grid * prob_z_less)
    
    return float(total_prob)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: 
    1. Start with a grid of support points.
    2. Optimize weights on this grid using gradient ascent / L-BFGS-B.
    3. Optionally optimize support points as well (though this is harder).
    4. Return the best valid measure found.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, keep 10s margin

    def project_simplex(v):
        """Project v onto the probability simplex."""
        n = len(v)
        if n == 0:
            return v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
        if len(rho) == 0:
            return np.ones(n) / n
        rho = rho[-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta, 0)

    def neg_score_weights(p, support):
        """Negative score for minimization."""
        w = np.asarray(p, dtype=np.float64)
        if np.any(w < -1e-9) or np.sum(w) <= 1e-9:
            return 100.0
        score = _compute_P(support, w)
        return -score

    def neg_score_and_grad(p, support):
        """Negative score and gradient for L-BFGS-B."""
        w = np.asarray(p, dtype=np.float64)
        n = len(support)
        if np.any(w < -1e-9) or np.sum(w) <= 1e-9:
            return 100.0, np.zeros(n)
        
        # Compute score
        score = _compute_P(support, w)
        
        # Compute gradient numerically or analytically?
        # Analytical gradient is complex. Let's use finite differences for robustness
        # or derive it.
        # P = sum_{k,l} w_k w_l C_kl(w)
        # dP/dw_m = sum_{k,l} (delta_km w_l + w_k delta_lm) C_kl + sum_{k,l} w_k w_l dC_kl/dw_m
        
        # For efficiency and simplicity in this context, let's try a sparse finite difference
        # or just use the numerical gradient provided by L-BFGS-B if we don't provide one.
        # But providing gradient is much faster.
        
        # Let's derive the gradient.
        # P = E[ P(X1+X2 < 2X4-X3 | X3, X4) ]
        # Let Z = X1+X2.
        # C_kl = P(Z < 2s_l - s_k) = sum_{i,j: s_i+s_j < 2s_l-s_k} w_i w_j
        # dC_kl/dw_m = sum_{j: s_m+s_j < 2s_l-s_k} w_j + sum_{i: s_i+s_m < 2s_l-s_k} w_i
        # Note: if i=m and j=m, it's counted once in the sum? No, sum is over pairs (i,j).
        # d/dw_m (sum_{i,j} I(s_i+s_j < T) w_i w_j) = sum_{j} I(s_m+s_j < T) w_j + sum_{i} I(s_i+s_m < T) w_i
        # Since the sum is symmetric, this is 2 * sum_{j: s_m+s_j < T} w_j - I(2s_m < T) w_m ?
        # Wait. sum_{i,j} w_i w_j f(i,j). d/dw_m = sum_j f(m,j) w_j + sum_i f(i,m) w_i.
        # If f is symmetric (which it is, s_i+s_j), then sum_j f(m,j) w_j = sum_i f(i,m) w_i.
        # So dC_kl/dw_m = 2 * sum_{j: s_m+s_j < 2s_l-s_k} w_j - (if m is counted twice? No.)
        # Let's check: sum_{i,j} w_i w_j. Term where i=m is w_m w_j. Term where j=m is w_i w_m.
        # If i=m and j=m, it's w_m^2. In the derivative, we get w_m (from i=m term) + w_m (from j=m term) = 2w_m.
        # The formula 2 * sum_j I(s_m+s_j < T) w_j counts j=m twice if 2s_m < T.
        # So yes, 2 * sum_{j: s_m+s_j < T} w_j is correct.
        
        # P = sum_{k,l} w_k w_l C_kl
        # dP/dw_m = sum_{k,l} (delta_km w_l + w_k delta_lm) C_kl + sum_{k,l} w_k w_l dC_kl/dw_m
        # = sum_l w_l C_ml + sum_k w_k C_km + sum_{k,l} w_k w_l dC_kl/dw_m
        
        # This requires computing sums of w_l C_ml etc.
        # Let's precompute C matrix (n x n) and its derivative contributions.
        
        s = support.astype(np.float64)
        w_norm = w / w.sum()
        
        # Compute C matrix
        s_sums = s[:, None] + s[None, :]
        sort_idx = np.argsort(s_sums.flatten(), kind='stable')
        s_sorted = s_sums.flatten()[sort_idx]
        
        # We need dC_kl/dw_m for all m.
        # dC_kl/dw_m = 2 * sum_{j: s_m+s_j < 2s_l-s_k} w_j
        # Let's precompute for each m, the vector of weights of j such that s_m+s_j < t.
        # This is tricky to vectorize for all m, k, l.
        
        # Given time constraints, let's use finite difference gradient for now.
        # It's O(n) evaluations of _compute_P.
        # For n=64, 64 evaluations per step. _compute_P is O(n^2 log n).
        # 64 * 64^2 * log(64) ~ 64 * 4096 * 6 ~ 1.5M ops.
        # L-BFGS-B might take ~100 steps. 150M ops. Should be fast enough.
        
        eps = 1e-6
        grad = np.zeros(n)
        for i in range(n):
            w_plus = w.copy()
            w_plus[i] += eps
            # Re-normalize? No, the function handles normalization internally if we pass raw weights?
            # _compute_P normalizes internally.
            f_plus = _compute_P(support, w_plus)
            
            w_minus = w.copy()
            w_minus[i] -= eps
            if w_minus[i] < -1e-9:
                w_minus[i] = 0.0
            f_minus = _compute_P(support, w_minus)
            
            grad[i] = (f_plus - f_minus) / (2.0 * eps)
            
        return -score, -grad

    # Initial setup
    # Start with a grid of 64 points on [0, 10]
    N = 64
    support = np.linspace(0.0, 10.0, N)
    # Initial weights: uniform
    w = np.ones(N) / N
    
    best_score = _compute_P(support, w)
    best_w = w.copy()
    best_support = support.copy()
    
    # Optimization loop
    # Use L-BFGS-B to maximize score (minimize -score)
    # We constrain w_i >= 0. We don't strictly constrain sum(w)=1 because _compute_P normalizes,
    # but it's better to work on the simplex or just let it scale.
    # Actually, if we don't constrain sum, the scale doesn't matter.
    # So we can just constrain w_i >= 0.
    
    bounds = [(0.0, None) for _ in range(N)]
    
    # Run optimization
    try:
        res = minimize(
            neg_score_weights,
            w,
            args=(support,),
            method='L-BFGS-B',
            jac=neg_score_and_grad,
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        
        if res.success or res.fun < -best_score:
            w_opt = res.x
            if np.sum(w_opt) > 1e-9:
                score_opt = _compute_P(support, w_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_w = w_opt.copy()
    except Exception as e:
        pass

    # Refine support points?
    # This is harder. We can try a local search on support points.
    # But let's first see if weight optimization alone is good.
    
    # Try a few different grid sizes and ranges
    grids = [
        (32, 0.0, 5.0),
        (64, 0.0, 10.0),
        (64, 0.0, 20.0),
        (128, 0.0, 10.0),
        (128, 0.0, 50.0)
    ]
    
    for N_grid, s_min, s_max in grids:
        if time.time() > deadline - 30.0:
            break
            
        support_g = np.linspace(s_min, s_max, N_grid)
        w_g = np.ones(N_grid) / N_grid
        
        # Quick check
        score_g = _compute_P(support_g, w_g)
        if score_g > best_score:
            best_score = score_g
            best_w = w_g.copy()
            best_support = support_g.copy()
            
        # Optimize weights on this grid
        bounds_g = [(0.0, None) for _ in range(N_grid)]
        
        try:
            res = minimize(
                neg_score_weights,
                w_g,
                args=(support_g,),
                method='L-BFGS-B',
                jac=neg_score_and_grad,
                bounds=bounds_g,
                options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-9}
            )
            
            if res.success or res.fun < -best_score:
                w_opt = res.x
                if np.sum(w_opt) > 1e-9:
                    score_opt = _compute_P(support_g, w_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_w = w_opt.copy()
                        best_support = support_g.copy()
        except Exception:
            pass

    # Local refinement on support points
    if time.time() < deadline - 60.0:
        current_support = best_support.copy()
        current_weights = best_w.copy()
        current_score = best_score
        n = len(current_support)
        
        # Simple local search on support
        step = 0.1
        for _ in range(20):
            if time.time() > deadline - 30.0:
                break
            improved = False
            for i in range(n):
                for delta in [-step, step]:
                    new_support = current_support.copy()
                    new_support[i] += delta
                    if np.any(new_support < -1e-9):
                        continue
                    try:
                        score = _compute_P(new_support, current_weights)
                        if score > current_score + 1e-10:
                            current_support = new_support
                            current_score = score
                            improved = True
                    except:
                        pass
            if not improved:
                step *= 0.5
                if step < 1e-4:
                    break
            
            if current_score > best_score:
                best_score = current_score
                best_support = current_support.copy()
                best_weights = current_weights.copy()
            else:
                # If no improvement in this round, maybe reset step?
                pass

    # Final validation
    if best_score > 0 and len(best_support) >= 1:
        if np.any(best_support < -1e-9):
            best_support = np.clip(best_support, 0, None)
        if np.any(best_weights < -1e-9):
            best_weights = np.clip(best_weights, 0, None)
        if best_weights.sum() <= 1e-9:
            # Fallback
            best_support = np.array([0.0, 1.0])
            best_weights = np.array([0.5, 0.5])

    # Normalize weights
    total_w = best_weights.sum()
    if total_w > 1e-9:
        best_weights = best_weights / total_w

    return best_support.astype(np.float64), best_weights.astype(np.float64)

# EVOLVE-BLOCK-END
