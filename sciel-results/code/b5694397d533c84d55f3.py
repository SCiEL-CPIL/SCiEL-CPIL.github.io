# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1..X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill

    def compute_P(support, weights):
        """
        Compute P[X1+X2+X3 < 2*X4] for a discrete measure.
        Uses vectorized numpy for speed: for each (k,l) pair, count (i,j) with s_i+s_j < 2*s_l - s_k.
        """
        n = len(support)
        if n == 0:
            return 0.0
        
        # Precompute pairwise sums: S[i,j] = support[i] + support[j]
        S = support[:, None] + support[None, :]  # shape (n, n)
        # Precompute pairwise weights: W[i,j] = weights[i] * weights[j]
        W = weights[:, None] * weights[None, :]  # shape (n, n)
        
        # We need: P = sum_{k,l} w_k * w_l * (sum_{i,j: S[i,j] < 2*s_l - s_k} W[i,j])
        # Vectorize over (k, l): threshold[k,l] = 2*s_l - s_k
        # threshold shape: (n, n) where threshold[k,l] = 2*support[l] - support[k]
        threshold = 2.0 * support[None, :] - support[:, None]  # shape (n, n): [k, l]
        
        # For each (k,l), we need sum of W[i,j] where S[i,j] < threshold[k,l]
        # This is a 4D problem. Let's loop over k and vectorize over l, or use broadcasting.
        # Since n is small (<= 5), we can afford a loop over k.
        
        total_P = 0.0
        for k in range(n):
            w_k = weights[k]
            if w_k <= 0:
                continue
            # For fixed k, threshold_l = 2*support[l] - support[k] for l in 0..n-1
            thresholds_l = 2.0 * support - support[k]  # shape (n,)
            # For each l, count sum of W[i,j] where S[i,j] < thresholds_l[l]
            # Create mask: (n, n, n) -> [l, i, j] where S[i,j] < thresholds_l[l]
            # S is (n, n), thresholds_l is (n,)
            # mask[l, i, j] = S[i, j] < thresholds_l[l]
            mask = S[None, :, :] < thresholds_l[:, None, None]  # shape (n, n, n)
            # Sum W[i,j] over i,j for each l: weighted_sum[l] = sum_{i,j} W[i,j] * mask[l,i,j]
            weighted_sums = np.sum(W[None, :, :] * mask, axis=(1, 2))  # shape (n,)
            # Total contribution for this k: w_k * sum_l w_l * weighted_sums[l]
            total_P += w_k * np.dot(weights, weighted_sums)
        
        return total_P

    # --- Strategy: Structural change to a different design family ---
    # Instead of fixed support points {0, 1, M}, we optimize the support points AND weights jointly.
    # We use a "free support" approach with n points, starting from good initial guesses.
    # We use Nelder-Mead on the log-weights and support positions.
    
    def neg_P_general(params, n_points):
        """
        params: [log_w0, log_w1, ..., log_w_{n-1}, s0, s1, ..., s_{n-1}]
        Weights are normalized from exp(log_w).
        Support is sorted.
        """
        log_w = params[:n_points]
        s = params[n_points:]
        
        # Support must be non-negative
        if np.any(s < -1e-12):
            return 1e10
        
        w = np.exp(log_w)
        w_sum = w.sum()
        if w_sum < 1e-12:
            return 1e10
        w = w / w_sum
        
        # Sort support and corresponding weights
        sort_idx = np.argsort(s)
        s_sorted = s[sort_idx]
        w_sorted = w[sort_idx]
        
        # Remove duplicates? If two supports are very close, merge them.
        # For simplicity, just compute P.
        P = compute_P(s_sorted, w_sorted)
        return -P

    # --- Initial Guesses ---
    # Based on the inspiration, the optimal 2-point is around p=0.5, M=1.
    # But 3-point and 4-point can do better.
    # Let's try a few different structural initializations.
    
    best_P = -1.0
    best_support = None
    best_weights = None
    
    # Helper to update best
    def update_best(s, w):
        nonlocal best_P, best_support, best_weights
        if len(s) == 0:
            return
        w = w / w.sum()
        P = compute_P(s, w)
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()

    # --- Phase 1: Analytic 2-point optimization (fast, gives good baseline) ---
    def two_point_P(p, L=1.0):
        """P[X1+X2+X3 < 2*X4] for 2-point {0, L} with mass p on 0."""
        # X4 = 0: need sum < 0 (impossible)
        # X4 = L: need sum < 2L. Sum is 0, L, 2L, 3L.
        # P[sum < 2L] = P[sum=0] + P[sum=L] = p^3 + 3*p^2*(1-p)
        return (1.0 - p) * (p**3 + 3.0 * p**2 * (1.0 - p))
    
    # Golden section search for p in (0,1)
    a, b = 0.01, 0.99
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = two_point_P(c)
    fd = two_point_P(d)
    for _ in range(100):
        if fc < fd:
            a = c; c = d; fc = fd
            d = a + gr * (b - a); fd = two_point_P(d)
        else:
            b = d; d = c; fd = fc
            c = b - gr * (b - a); fc = two_point_P(c)
    p_opt = (a + b) / 2.0
    update_best(np.array([0.0, 1.0]), np.array([p_opt, 1.0 - p_opt]))
    
    # --- Phase 2: Local optimization on 3-point and 4-point free supports ---
    # We will use scipy.optimize.minimize (Nelder-Mead) on the parameters.
    # To avoid dependency issues or slow imports, we can implement a simple Nelder-Mead
    # or just use scipy if available. The prompt says SciPy is available.
    
    from scipy.optimize import minimize
    
    # 3-point optimization
    # Parameters: log_w0, log_w1, log_w2, s0, s1, s2
    # Constraint: s0=0 is often optimal. Let's fix s0=0 to reduce dimensionality.
    # So params: log_w0, log_w1, log_w2, s1, s2
    def neg_P_3point_fixed0(params):
        log_w = params[:3]
        s1, s2 = params[3], params[4]
        if s1 < 0 or s2 < 0 or s2 < s1:
            return 1e10
        s = np.array([0.0, s1, s2])
        w = np.exp(log_w)
        w = w / w.sum()
        P = compute_P(s, w)
        return -P

    # Initial guesses for 3-point
    # From previous knowledge: {0, 1, M} with M ~ 2-3.
    # Let's try a few starts.
    inits_3 = [
        [0.0, 0.0, 0.0, 1.0, 2.0],  # uniform weights, s={0,1,2}
        [np.log(0.5), np.log(0.3), np.log(0.2), 1.0, 3.0],
        [np.log(0.4), np.log(0.4), np.log(0.2), 0.5, 2.0],
        [np.log(0.6), np.log(0.2), np.log(0.2), 1.0, 1.5],
    ]
    
    for init in inits_3:
        if time.time() > deadline - 60.0:
            break
        try:
            res = minimize(
                neg_P_3point_fixed0,
                init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < -best_P:
                log_w = res.x[:3]
                s1, s2 = res.x[3], res.x[4]
                s = np.array([0.0, s1, s2])
                w = np.exp(log_w)
                w = w / w.sum()
                update_best(s, w)
        except Exception:
            pass

    # 4-point optimization
    # Params: log_w0..log_w3, s1, s2, s3 (s0=0 fixed)
    def neg_P_4point_fixed0(params):
        log_w = params[:4]
        s1, s2, s3 = params[4], params[5], params[6]
        if s1 < 0 or s2 < 0 or s3 < 0 or s2 < s1 or s3 < s2:
            return 1e10
        s = np.array([0.0, s1, s2, s3])
        w = np.exp(log_w)
        w = w / w.sum()
        P = compute_P(s, w)
        return -P

    inits_4 = [
        [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 2.0],
        [np.log(0.4), np.log(0.3), np.log(0.2), np.log(0.1), 0.5, 1.0, 2.0],
        [np.log(0.3), np.log(0.3), np.log(0.3), np.log(0.1), 0.2, 1.0, 3.0],
    ]
    
    for init in inits_4:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(
                neg_P_4point_fixed0,
                init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < -best_P:
                log_w = res.x[:4]
                s1, s2, s3 = res.x[4], res.x[5], res.x[6]
                s = np.array([0.0, s1, s2, s3])
                w = np.exp(log_w)
                w = w / w.sum()
                update_best(s, w)
        except Exception:
            pass

    # --- Phase 3: Local refinement of the best found support/weights ---
    # If we have a good candidate, do a finer local search on the support positions and weights.
    if best_support is not None and time.time() < deadline - 10.0:
        n = len(best_support)
        # Ensure s0 is 0? Not necessarily, but likely.
        # We will optimize all support points and log weights.
        
        def neg_P_full(params):
            log_w = params[:n]
            s = params[n:]
            if np.any(s < -1e-12):
                return 1e10
            # Sort
            sort_idx = np.argsort(s)
            s_sorted = s[sort_idx]
            w = np.exp(log_w)
            w = w / w.sum()
            w_sorted = w[sort_idx]
            P = compute_P(s_sorted, w_sorted)
            return -P
        
        # Initial params
        init_log_w = np.log(np.maximum(best_weights, 1e-12))
        init_s = best_support.copy()
        init_params = np.concatenate([init_log_w, init_s])
        
        try:
            res = minimize(
                neg_P_full,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if res.fun < -best_P:
                log_w = res.x[:n]
                s = res.x[n:]
                sort_idx = np.argsort(s)
                s_sorted = s[sort_idx]
                w = np.exp(log_w)
                w = w / w.sum()
                w_sorted = w[sort_idx]
                update_best(s_sorted, w_sorted)
        except Exception:
            pass

    # --- Final Output ---
    if best_support is None:
        # Fallback
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    
    # Clean up: remove points with negligible weight
    mask = best_weights > 1e-10
    best_support = best_support[mask]
    best_weights = best_weights[mask]
    
    # Normalize
    best_weights = best_weights / best_weights.sum()
    
    # Validation
    assert len(best_support) == len(best_weights)
    assert len(best_support) >= 1
    assert np.all(np.isfinite(best_support))
    assert np.all(np.isfinite(best_weights))
    assert np.all(best_support >= -1e-9)
    assert np.all(best_weights >= -1e-9)
    assert np.sum(best_weights) > 1e-9
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)

# EVOLVE-BLOCK-END
