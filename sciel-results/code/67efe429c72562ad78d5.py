# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0


def _remaining():
    return _DEADLINE - (time.time() - _START_TIME) - 10.0


def _prob_3point(p, q, M):
    """
    Compute P[X1+X2+X3 < 2*X4] for distribution at {0, 1, M}.
    p = P(X=0), q = P(X=1), r = P(X=M) = 1-p-q.
    Assumes M > 1.
    """
    r = 1.0 - p - q
    if r < -1e-12 or p < 0 or q < 0:
        return 0.0
    r = max(r, 0.0)
    
    if M > 1.0 + 1e-9:
        # Case X4 = 1: Need X1+X2+X3 < 2.
        # Possible sums: 0 (p^3), 1 (3*p^2*q).
        # M > 1 implies any term with M makes sum >= M > 1.
        # If M is close to 1, e.g. M=1+eps, sum could be M+0+0 = M < 2?
        # Wait. If M=1.1, sum=1.1 < 2.
        # So we need to be careful with the "M > 1" assumption vs "M >= 2".
        # Let's check the condition strictly: s_i + s_j + s_k < 2.
        # If M >= 2, then any sum involving M is >= 2, so not < 2.
        # If 1 < M < 2, then M < 2, so M+0+0 is a valid sum.
        
        # Let's derive generally for M > 1.
        # Sums < 2:
        # 0: p^3
        # 1: 3*p^2*q
        # M: 3*p^2*r (if M < 2)
        # 1+M: 3*p*q*r (if 1+M < 2 => M < 1, false)
        # So if 1 < M < 2:
        # prob_x4_1 = p^3 + 3*p^2*q + 3*p^2*r
        
        # If M >= 2:
        # prob_x4_1 = p^3 + 3*p^2*q
        
        if M < 2.0:
            prob_x4_1 = p**3 + 3*p**2*q + 3*p**2*r
        else:
            prob_x4_1 = p**3 + 3*p**2*q

        # Case X4 = M: Need X1+X2+X3 < 2M.
        # Sums possible:
        # 0, 1, M, M+1, 2M, 2M+1, 3M
        # We need sum < 2M.
        # 0 < 2M (always true for M>0)
        # 1 < 2M (always true for M>0.5)
        # M < 2M (always true for M>0)
        # M+1 < 2M => 1 < M (true)
        # 2M < 2M (False, strict inequality)
        # So valid sums are 0, 1, M, M+1.
        
        # Coeffs:
        # 0: p^3
        # 1: 3*p^2*q
        # M: 3*p^2*r
        # M+1: 3*p*q^2
        
        prob_x4_M = p**3 + 3*p**2*q + 3*p**2*r + 3*p*q**2
        
        total = q * prob_x4_1 + r * prob_x4_M
        return total
    else:
        # Fallback to brute force for M <= 1 (shouldn't happen in optimization)
        support = np.array([0.0, 1.0, M])
        weights = np.array([p, q, r])
        return _prob_brute(support, weights)


def _prob_brute(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] by direct enumeration."""
    n = len(support)
    w = weights / weights.sum()
    
    total_prob = 0.0
    for idx4 in range(n):
        w4 = w[idx4]
        threshold = 2.0 * support[idx4]
        p_sum = 0.0
        for idx1 in range(n):
            for idx2 in range(n):
                for idx3 in range(n):
                    if support[idx1] + support[idx2] + support[idx3] < threshold:
                        p_sum += w[idx1] * w[idx2] * w[idx3]
        total_prob += w4 * p_sum
    
    return total_prob


def _optimize_3point():
    """Optimize the 3-point distribution {0, 1, M}."""
    
    def neg_prob(params):
        p, q, M = params
        if p < 0 or q < 0 or p + q > 1 or M < 0:
            return 1.0
        return -_prob_3point(p, q, M)
    
    best_prob = 0.0
    best_params = [0.5, 0.3, 10.0]
    
    # Try multiple starting points for Nelder-Mead
    start_points = [
        [0.5, 0.3, 10.0],
        [0.6, 0.2, 5.0],
        [0.4, 0.4, 2.0],
        [0.7, 0.1, 1.5],
        [0.3, 0.5, 8.0],
        [0.2, 0.6, 3.0],
    ]
    
    for sp in start_points:
        if _remaining() < 5.0:
            break
        try:
            res = minimize(neg_prob, sp, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < 0:
                prob = -res.fun
                if prob > best_prob:
                    best_prob = prob
                    best_params = res.x
        except Exception:
            pass
            
    # Also try a grid search for M if needed, or just trust NM
    # Let's do a quick 1D scan for M with optimal p, q from previous best?
    # No, NM should handle it.
    
    p, q, M = best_params
    r = 1.0 - p - q
    if r < 0:
        r = 0.0
    # Normalize just in case of numerical drift
    total = p + q + r
    p, q, r = p/total, q/total, r/total
    
    support = np.array([0.0, 1.0, M])
    weights = np.array([p, q, r])
    return support, weights, best_prob


def _optimize_4point():
    """
    Optimize a 4-point distribution {0, 1, M1, M2}.
    Weights: p0, p1, pM1, pM2.
    This is 6 parameters.
    """
    
    def prob_4point(params):
        p0, p1, pM1, pM2, M1, M2 = params
        if p0 < 0 or p1 < 0 or pM1 < 0 or pM2 < 0:
            return 0.0
        total_w = p0 + p1 + pM1 + pM2
        if total_w <= 0:
            return 0.0
        # Normalize
        p0 /= total_w
        p1 /= total_w
        pM1 /= total_w
        pM2 /= total_w
        
        if M1 <= 1.0 or M2 <= M1:
            return 0.0
            
        support = np.array([0.0, 1.0, M1, M2])
        weights = np.array([p0, p1, pM1, pM2])
        return _prob_brute(support, weights)
    
    def neg_prob(params):
        return -prob_4point(params)
    
    best_prob = 0.0
    best_params = None
    
    start_points = [
        [0.4, 0.3, 0.2, 0.1, 2.0, 10.0],
        [0.3, 0.3, 0.2, 0.2, 1.5, 5.0],
        [0.5, 0.2, 0.2, 0.1, 3.0, 20.0],
    ]
    
    for sp in start_points:
        if _remaining() < 10.0:
            break
        try:
            res = minimize(neg_prob, sp, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < 0:
                prob = -res.fun
                if prob > best_prob:
                    best_prob = prob
                    best_params = res.x
        except Exception:
            pass
            
    if best_params is not None:
        p0, p1, pM1, pM2, M1, M2 = best_params
        total_w = p0 + p1 + pM1 + pM2
        if total_w > 0:
            p0 /= total_w
            p1 /= total_w
            pM1 /= total_w
            pM2 /= total_w
            support = np.array([0.0, 1.0, M1, M2])
            weights = np.array([p0, p1, pM1, pM2])
            return support, weights, best_prob
            
    return None, None, 0.0


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    
    best_support = None
    best_weights = None
    best_prob = 0.0
    
    # 1. Try 3-point optimization (fast and robust)
    s3, w3, p3 = _optimize_3point()
    if s3 is not None and p3 > best_prob:
        best_support, best_weights, best_prob = s3, w3, p3
        
    # 2. Try 4-point optimization if time permits
    if _remaining() > 30.0:
        s4, w4, p4 = _optimize_4point()
        if s4 is not None and p4 > best_prob:
            best_support, best_weights, best_prob = s4, w4, p4
            
    # Fallback if something went wrong
    if best_support is None:
        best_support = np.array([0.0, 1.0, 10.0])
        best_weights = np.array([0.5, 0.3, 0.2])
        
    # Ensure validity
    best_support = best_support.astype(np.float64)
    best_weights = best_weights.astype(np.float64)
    
    # Clip small negative values
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Ensure weights sum > 1e-9
    if best_weights.sum() <= 1e-9:
        best_weights = np.ones_like(best_weights) / len(best_weights)
        
    return best_support, best_weights

# EVOLVE-BLOCK-END
