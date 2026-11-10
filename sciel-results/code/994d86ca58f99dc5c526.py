import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from itertools import product


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.

    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 600s kill

    def compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] for given support and weights."""
        n = len(support)
        if n == 0:
            return 0.0
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        w = weights / w_sum
        s = support
        
        # Precompute all pairwise sums and their weights
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        pair_sums = s[i_idx] + s[j_idx]
        pair_weights = w[i_idx] * w[j_idx]
        
        # Sort pair_sums for efficient threshold queries
        flat_sums = pair_sums.flatten()
        flat_weights = pair_weights.flatten()
        sort_idx = np.argsort(flat_sums)
        sorted_sums = flat_sums[sort_idx]
        sorted_weights = flat_weights[sort_idx]
        cum_weights = np.cumsum(sorted_weights)
        
        P = 0.0
        for l in range(n):
            target = 2.0 * s[l]
            thresholds = target - s
            idxs = np.searchsorted(sorted_sums, thresholds, side='left')
            inner_sums = np.zeros(n)
            mask = idxs > 0
            inner_sums[mask] = cum_weights[idxs[mask] - 1]
            P += w[l] * np.sum(w * inner_sums)
            
        return P

    def neg_P(params, n):
        """Negative P for minimization. params = [support..., weights...]"""
        support = np.maximum(params[:n], 0.0)
        weights = np.maximum(params[n:], 0.0)
        w_sum = np.sum(weights)
        if w_sum < 1e-15:
            return 1.0
        weights = weights / w_sum
        return -compute_P(support, weights)

    def optimize_weights_for_support(support, maxiter=500):
        """Given fixed support, optimize weights to maximize P."""
        n = len(support)
        
        def neg_P_w(w_params):
            w = np.maximum(w_params, 0.0)
            w_sum = np.sum(w)
            if w_sum < 1e-15:
                return 1.0
            w = w / w_sum
            return -compute_P(support, w)
        
        best_w = None
        best_score = -1.0
        
        # Try multiple starting points for weights
        starts = [
            np.ones(n) / n,
            np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625][:n]),
            np.array([0.6, 0.2, 0.1, 0.05, 0.03, 0.01, 0.005, 0.0025, 0.00125, 0.000625][:n]),
        ]
        
        for w0 in starts:
            if len(w0) != n:
                continue
            try:
                res = minimize(neg_P_w, w0, method='SLSQP', 
                             bounds=[(0.0, 1.0)] * n,
                             options={'maxiter': maxiter, 'ftol': 1e-12})
                if res.fun < best_score:
                    best_score = res.fun
                    best_w = res.x.copy()
            except Exception:
                pass
        
        if best_w is None:
            best_w = np.ones(n) / n
        
        best_w = np.maximum(best_w, 0.0)
        w_sum = np.sum(best_w)
        if w_sum > 1e-9:
            best_w = best_w / w_sum
        
        return best_w, -best_score

    def optimize_support_for_weights(weights, maxiter=500):
        """Given fixed weights, optimize support to maximize P."""
        n = len(weights)
        w = weights / np.sum(weights)
        
        def neg_P_s(s_params):
            s = np.maximum(s_params, 0.0)
            return -compute_P(s, w)
        
        best_s = None
        best_score = -1.0
        
        # Try multiple starting points for support
        starts = [
            np.arange(n, dtype=np.float64),
            np.linspace(0, 10, n),
            np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0][:n]),
            np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0][:n]),
        ]
        
        for s0 in starts:
            if len(s0) != n:
                continue
            try:
                res = minimize(neg_P_s, s0, method='Nelder-Mead',
                             options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_score:
                    best_score = res.fun
                    best_s = res.x.copy()
            except Exception:
                pass
        
        if best_s is None:
            best_s = np.arange(n, dtype=np.float64)
        
        best_s = np.maximum(best_s, 0.0)
        return best_s, -best_score

    def coordinate_descent(support_init, weights_init, n, max_rounds=20):
        """Alternate between optimizing support and weights."""
        s = support_init.copy()
        w = weights_init.copy()
        
        for round_idx in range(max_rounds):
            if time.time() > deadline - 10.0:
                break
            
            # Optimize weights given support
            w_new, score_w = optimize_weights_for_support(s, maxiter=200)
            w = w_new
            
            # Optimize support given weights
            s_new, score_s = optimize_support_for_weights(w, maxiter=200)
            s = s_new
        
        final_score = compute_P(s, w)
        return s, w, final_score

    best_score = 0.0
    best_support = None
    best_weights = None

    def try_candidate(s, w):
        nonlocal best_score, best_support, best_weights
        if len(s) != len(w) or len(s) == 0:
            return
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return
        w_sum = np.sum(w)
        if w_sum <= 1e-9:
            return
        w_norm = w / w_sum
        score = compute_P(s, w_norm)
        if score > best_score:
            best_score = score
            best_support = s.copy()
            best_weights = w_norm.copy()

    # Analytical 2-point solution
    # f(p) = p(1-p)^2(1+2p), maximize over p in [0,1]
    def f_2point(p):
        return p * (1-p)**2 * (1+2*p)
    
    res_2p = minimize_scalar(lambda p: -f_2point(p), bounds=(0.0, 1.0), method='bounded')
    p_opt = res_2p.x
    M_vals = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    for M in M_vals:
        s = np.array([0.0, M])
        w = np.array([1-p_opt, p_opt])
        try_candidate(s, w)
    
    # Try n=3,4,5 with coordinate descent
    for n in [3, 4, 5]:
        if time.time() > deadline - 120.0:
            break
        
        # Multiple initial configurations
        inits = []
        
        # Linear support
        s0 = np.linspace(0, 5, n)
        w0 = np.ones(n) / n
        inits.append((s0, w0))
        
        # Geometric support
        s0 = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0][:n])
        w0 = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625][:n])
        w0 = w0 / w0.sum()
        inits.append((s0, w0))
        
        # Clustered near 0 and far out
        s0 = np.concatenate([[0.0, 0.1, 0.2, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0][:n]])
        w0 = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.015, 0.005][:n])
        w0 = w0 / w0.sum()
        inits.append((s0, w0))
        
        # Symmetric-ish
        s0 = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0][:n])
        w0 = np.array([0.4, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005][:n])
        w0 = w0 / w0.sum()
        inits.append((s0, w0))
        
        for s_init, w_init in inits:
            if time.time() > deadline - 60.0:
                break
            try:
                s_opt, w_opt, score = coordinate_descent(s_init, w_init, n, max_rounds=15)
                try_candidate(s_opt, w_opt)
            except Exception:
                pass

    # Try n=2 with finer search over M
    if time.time() < deadline - 60.0:
        M_grid = np.logspace(-1, 2, 50)
        for M in M_grid:
            s = np.array([0.0, M])
            w = np.array([1-p_opt, p_opt])
            try_candidate(s, w)

    # Try n=3 with a dedicated Nelder-Mead on all parameters
    if time.time() < deadline - 60.0:
        n = 3
        def neg_P_all(params):
            s = np.maximum(params[:n], 0.0)
            w = np.maximum(params[n:], 0.0)
            w_sum = np.sum(w)
            if w_sum < 1e-15:
                return 1.0
            w = w / w_sum
            return -compute_P(s, w)
        
        inits = [
            np.concatenate([np.array([0.0, 1.0, 2.0]), np.array([0.5, 0.3, 0.2])]),
            np.concatenate([np.array([0.0, 0.5, 1.0]), np.array([0.6, 0.3, 0.1])]),
            np.concatenate([np.array([0.0, 2.0, 5.0]), np.array([0.4, 0.35, 0.25])]),
        ]
        
        for init in inits:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(neg_P_all, init, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
                s_opt = np.maximum(res.x[:n], 0.0)
                w_opt = np.maximum(res.x[n:], 0.0)
                try_candidate(s_opt, w_opt)
            except Exception:
                pass

    # Try n=4 with Nelder-Mead
    if time.time() < deadline - 60.0:
        n = 4
        def neg_P_all(params):
            s = np.maximum(params[:n], 0.0)
            w = np.maximum(params[n:], 0.0)
            w_sum = np.sum(w)
            if w_sum < 1e-15:
                return 1.0
            w = w / w_sum
            return -compute_P(s, w)
        
        inits = [
            np.concatenate([np.array([0.0, 1.0, 2.0, 4.0]), np.array([0.4, 0.3, 0.2, 0.1])]),
            np.concatenate([np.array([0.0, 0.5, 1.0, 2.0]), np.array([0.5, 0.25, 0.15, 0.1])]),
            np.concatenate([np.array([0.0, 1.0, 3.0, 6.0]), np.array([0.45, 0.3, 0.15, 0.1])]),
        ]
        
        for init in inits:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(neg_P_all, init, method='Nelder-Mead',
                             options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10})
                s_opt = np.maximum(res.x[:n], 0.0)
                w_opt = np.maximum(res.x[n:], 0.0)
                try_candidate(s_opt, w_opt)
            except Exception:
                pass

    # If no valid result, use fallback
    if best_support is None or best_score <= 0.0:
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        # Ensure validity
        if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
            n = 10
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n
        elif np.sum(best_weights) <= 1e-9:
            n = 10
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n

    return best_support, best_weights
