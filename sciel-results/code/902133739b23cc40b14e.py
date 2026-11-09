# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _compute_P(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] exactly for a discrete measure.
    Uses the formula: P = sum_{k,l} w_k*w_l * sum_{i,j: s_i+s_j < 2*s_l-s_k} w_i*w_j
    
    We vectorize using the pair-sum approach with a cumulative trick.
    """
    n = len(support)
    w = weights / weights.sum()
    
    # Build pair sum matrix: S[i,j] = s_i + s_j
    pair_sums = support[:, None] + support[None, :]
    pair_weights = w[:, None] * w[None, :]
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            mask = pair_sums < threshold
            inner = np.sum(pair_weights * mask)
            P += w[k] * w[l] * inner
    return P


def _compute_P_fast(support, weights):
    """
    Fast computation using sorted support and prefix sums for the pair-sum CDF.
    P = sum_{k,l} w_k*w_l * F(2*s_l - s_k)
    where F(t) = P[X1+X2 < t] = sum_{i,j: s_i+s_j < t} w_i*w_j
    
    For sorted support, for fixed threshold t:
    For each i, find max j such that s_i + s_j < t, i.e., s_j < t - s_i.
    F(t) = sum_i w_i * (sum of w_j for j where s_j < t - s_i)
    """
    n = len(support)
    w = weights / weights.sum()
    w_prefix = np.concatenate([[0.0], np.cumsum(w)])
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            if threshold <= 0:
                continue
            inner = 0.0
            for i in range(n):
                target = threshold - support[i]
                if target <= 0:
                    break
                j_max = np.searchsorted(support, target, side='left') - 1
                if j_max >= 0:
                    inner += w[i] * w_prefix[j_max + 1]
            P += w[k] * w[l] * inner
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Structural approach: model the problem as an optimization over continuous
    parameters for small-n discrete distributions, using scipy.optimize.
    
    Key structural change: instead of grid search, use scipy's minimizer on
    the negative P function directly. We parameterize:
    1. Two-point: {0, M} with probability q at M
    2. Three-point: {0, a, M} with probabilities p0, p1, p2
    3. Four-point: {0, a, b, M} with geometric-ish spacing
    
    We use differential evolution for global search on each family, then
    polish with L-BFGS-B.
    """
    deadline = time.time() + 570
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def _update_best(P, support, weights):
        nonlocal best_score, best_support, best_weights
        if P > best_score:
            best_score = P
            best_support = support.copy()
            best_weights = weights.copy()
    
    # ===== Two-point: {0, M}, P(X=M)=q =====
    # P = q * (1-q)^2 * (1+2q)
    # Analytical optimum: q* = (1+sqrt(33))/16
    q_star = (1.0 + np.sqrt(33.0)) / 16.0
    P_two = q_star * (1 - q_star)**2 * (1 + 2 * q_star)
    _update_best(P_two, np.array([0.0, 1.0]), np.array([1 - q_star, q_star]))
    
    # Verify with exact computation
    sup2 = np.array([0.0, 1.0])
    wt2 = np.array([1 - q_star, q_star])
    P2_v = _compute_P(sup2, wt2)
    _update_best(P2_v, sup2, wt2)
    
    # ===== Three-point: {0, a, M} with M=1, a = ratio =====
    # Parameters: ratio (a/M), p1 (prob at a), p2 (prob at M), p0 = 1-p1-p2
    # P = f(ratio, p1, p2)
    # We search over ratio and optimize (p1, p2) with scipy.
    
    def _P_three_params(params):
        if time.time() > deadline:
            return 0.0
        ratio, p1, p2 = params
        if p1 < 0 or p2 < 0 or p1 + p2 > 1:
            return -1.0  # invalid
        p0 = 1.0 - p1 - p2
        if p0 < 0:
            return -1.0
        sup = np.array([0.0, ratio, 1.0])
        wt = np.array([p0, p1, p2])
        try:
            return _compute_P_fast(sup, wt)
        except Exception:
            return -1.0
    
    def _neg_P_three(params):
        return -_P_three_params(params)
    
    # Search over ratio values
    for ratio in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if time.time() > deadline:
            break
        # Optimize p1, p2 given ratio
        bounds = [(0.0, 0.9), (0.0, 0.9)]
        # Try multiple starting points
        best_local = -1.0
        for p1_0 in [0.1, 0.3, 0.5, 0.7]:
            for p2_0 in [0.1, 0.3, 0.5, 0.7]:
                x0 = np.array([p1_0, p2_0])
                if x0[0] + x0[1] > 0.95:
                    continue
                try:
                    res = minimize(_neg_P_three if False else lambda x: -_P_three_params(np.array([ratio, x[0], x[1]])), 
                                   x0, method='L-BFGS-B', bounds=bounds, 
                                   options={'maxiter': 200, 'ftol': 1e-12})
                    P_val = -res.fun
                    if P_val > best_local:
                        best_local = P_val
                        best_p1, best_p2 = res.x[0], res.x[1]
                except Exception:
                    continue
        
        if best_local > -0.5:
            p0 = 1.0 - best_p1 - best_p2
            sup = np.array([0.0, ratio, 1.0])
            wt = np.array([max(p0, 0), max(best_p1, 0), max(best_p2, 0)])
            wt = wt / wt.sum()
            P_val = _compute_P_fast(sup, wt)
            _update_best(P_val, sup, wt)
    
    # ===== Four-point: {0, a, b, 1} =====
    # Parameters: a, b, p1, p2, p3 (p0 = 1 - p1 - p2 - p3)
    # We use a coarse grid for (a, b) and optimize weights.
    
    def _P_four_params(params, a, b):
        p1, p2, p3 = params
        if p1 < 0 or p2 < 0 or p3 < 0 or p1 + p2 + p3 > 1:
            return -1.0
        p0 = 1.0 - p1 - p2 - p3
        if p0 < 0:
            return -1.0
        sup = np.array([0.0, a, b, 1.0])
        wt = np.array([p0, p1, p2, p3])
        try:
            return _compute_P_fast(sup, wt)
        except Exception:
            return -1.0
    
    for a in [0.05, 0.1, 0.2, 0.3, 0.5]:
        if time.time() > deadline:
            break
        for b in [0.5, 0.6, 0.7, 0.8, 0.9]:
            if time.time() > deadline:
                break
            if a >= b:
                continue
            bounds4 = [(0.0, 0.9), (0.0, 0.9), (0.0, 0.9)]
            best_local = -1.0
            for p1_0 in [0.1, 0.3, 0.5]:
                for p2_0 in [0.1, 0.3, 0.5]:
                    for p3_0 in [0.1, 0.3, 0.5]:
                        x0 = np.array([p1_0, p2_0, p3_0])
                        if x0.sum() > 0.9:
                            continue
                        try:
                            res = minimize(lambda x: -_P_four_params(x, a, b), 
                                           x0, method='L-BFGS-B', bounds=bounds4,
                                           options={'maxiter': 150, 'ftol': 1e-12})
                            P_val = -res.fun
                            if P_val > best_local:
                                best_local = P_val
                                best_p = res.x
                        except Exception:
                            continue
            if best_local > -0.5:
                p0 = 1.0 - best_p[0] - best_p[1] - best_p[2]
                sup = np.array([0.0, a, b, 1.0])
                wt = np.array([max(p0, 0), max(best_p[0], 0), max(best_p[1], 0), max(best_p[2], 0)])
                wt = wt / wt.sum()
                P_val = _compute_P_fast(sup, wt)
                _update_best(P_val, sup, wt)
    
    # ===== Five-point: geometric {0, 1/r, 1/r^2, 1/r^3, 1} =====
    for r in [2.0, 2.5, 3.0, 4.0]:
        if time.time() > deadline:
            break
        sup = np.array([0.0, 1.0/r, 1.0/r**2, 1.0/r**3, 1.0])
        sup = np.sort(sup)
        # Optimize weights
        def _P_five(wt):
            p = wt / wt.sum()
            return _compute_P_fast(sup, p)
        
        best_local = -1.0
        for init in [np.array([0.3, 0.2, 0.15, 0.1, 0.25]),
                     np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                     np.array([0.4, 0.1, 0.1, 0.1, 0.3]),
                     np.array([0.1, 0.1, 0.1, 0.3, 0.4])]:
            try:
                res = minimize(lambda x: -_P_five(np.exp(x)), 
                               np.log(init + 1e-10), method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12})
                P_val = -res.fun
                if P_val > best_local:
                    best_local = P_val
                    best_wt = np.exp(res.x)
            except Exception:
                continue
        if best_local > -0.5:
            best_wt = best_wt / best_wt.sum()
            P_val = _compute_P_fast(sup, best_wt)
            _update_best(P_val, sup, best_wt)
    
    # ===== Final validation =====
    if best_support is None or best_weights is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([1 - q_star, q_star], dtype=np.float64)
    
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    # Ensure validity
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    if best_weights.sum() <= 1e-9:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([1 - q_star, q_star], dtype=np.float64)
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
