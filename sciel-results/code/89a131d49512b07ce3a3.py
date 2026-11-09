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
    
    # --- Phase 2: Analytical 3-point optimization on fixed support {0, 1, M} ---
    # For support {0, 1, M} with weights (a, b, c) where a+b+c=1,
    # P[X1+X2+X3 < 2*X4] can be computed directly.
    # We do a 2D grid search over M and b (mass at 1), with a = 1-b-c and c determined.
    # Actually, we search over (b, c) and M, with a = 1-b-c >= 0.
    
    def P_3point(M, b, c):
        """Compute P for support {0, 1, M} with weights (a, b, c)."""
        a = 1.0 - b - c
        if a < -1e-15 or b < -1e-15 or c < -1e-15:
            return 0.0
        a = max(a, 0.0)
        b = max(b, 0.0)
        c = max(c, 0.0)
        s = np.array([0.0, 1.0, M])
        w = np.array([a, b, c])
        return compute_P(s, w)
    
    # Grid search over M and (b, c)
    # M ranges from just above 1 to some large value
    # b + c <= 1, b >= 0, c >= 0
    
    # Coarse grid first
    M_vals = np.linspace(1.01, 10.0, 200)
    bc_grid = 50  # grid for b and c
    
    best_M = 2.0
    best_b = 0.3
    best_c = 0.2
    best_P_grid = -1.0
    
    for M in M_vals:
        if time.time() > deadline - 120.0:
            break
        # For fixed M, search over b, c with b+c <= 1
        # Let t = b+c, then b = t*u, c = t*(1-u) for u in [0,1]
        # But simpler: loop over b from 0 to 1, c from 0 to 1-b
        for bi in range(bc_grid + 1):
            b_val = bi / bc_grid
            for ci in range(bc_grid + 1):
                c_val = ci / bc_grid * (1.0 - b_val)
                a_val = 1.0 - b_val - c_val
                if a_val < -1e-15:
                    continue
                P_val = P_3point(M, b_val, c_val)
                if P_val > best_P_grid:
                    best_P_grid = P_val
                    best_M = M
                    best_b = b_val
                    best_c = c_val
    
    # Update best with grid result
    if best_P_grid > best_P:
        a = 1.0 - best_b - best_c
        s = np.array([0.0, 1.0, best_M])
        w = np.array([a, best_b, best_c])
        update_best(s, w)
    
    # Fine grid around the best found
    if time.time() < deadline - 60.0:
        M_fine = np.linspace(max(1.01, best_M * 0.5), best_M * 2.0, 400)
        bc_fine = 150
        
        for M in M_fine:
            if time.time() > deadline - 60.0:
                break
            for bi in range(bc_fine + 1):
                b_val = bi / bc_fine
                for ci in range(bc_fine + 1):
                    c_val = ci / bc_fine * (1.0 - b_val)
                    a_val = 1.0 - b_val - c_val
                    if a_val < -1e-15:
                        continue
                    P_val = P_3point(M, b_val, c_val)
                    if P_val > best_P:
                        a = 1.0 - b_val - c_val
                        s = np.array([0.0, 1.0, M])
                        w = np.array([a, b_val, c_val])
                        update_best(s, w)
    
    # --- Phase 3: Even finer local refinement around the best 3-point solution ---
    if best_support is not None and len(best_support) == 3 and time.time() < deadline - 20.0:
        # Extract current best parameters
        s_best = best_support.copy()
        w_best = best_weights.copy()
        # s_best should be [0, 1, M] approximately
        M_cur = s_best[2]
        b_cur = w_best[1]
        c_cur = w_best[2]
        
        # Very fine grid
        M_very_fine = np.linspace(M_cur * 0.95, M_cur * 1.05, 200)
        bc_very_fine = 300
        
        for M in M_very_fine:
            if time.time() > deadline - 20.0:
                break
            for bi in range(bc_very_fine + 1):
                b_val = b_cur * 0.8 + (bi / bc_very_fine) * b_cur * 0.4
                if b_val < 0 or b_val > 1:
                    continue
                for ci in range(bc_very_fine + 1):
                    c_val = c_cur * 0.8 + (ci / bc_very_fine) * c_cur * 0.4
                    if c_val < 0 or b_val + c_val > 1 + 1e-15:
                        continue
                    a_val = 1.0 - b_val - c_val
                    if a_val < -1e-15:
                        continue
                    P_val = P_3point(M, b_val, c_val)
                    if P_val > best_P:
                        a = 1.0 - b_val - c_val
                        s = np.array([0.0, 1.0, M])
                        w = np.array([a, b_val, c_val])
                        update_best(s, w)

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
