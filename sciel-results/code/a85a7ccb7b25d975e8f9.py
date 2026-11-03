import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the score: L2^2 / (L1 * Sup(autoconv))
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def construct():
    start_time = time.time()
    deadline = 140.0

    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def find_sidon_set(N):
        S = [0]
        for i in range(1, N):
            candidate = S + [i]
            if is_sidon(candidate):
                S = candidate
        return S

    sidon_set = find_sidon_set(N)
    K = len(sidon_set)
    sorted_sidon = sorted(sidon_set)

    # Initialize with uniform weights on Sidon set
    best_h = np.zeros(N)
    best_h[sorted_sidon] = 1.0 / K
    best_score = _score_raw(best_h)

    # LP formulation: Fix L1=1, minimize peak autoconvolution
    # The autoconvolution at lag k is sum_i h_i * h_{i+k}
    # We want to minimize max_k (h*h)_k subject to sum(h)=1, h>=0
    # This is: min t s.t. (h*h)_k <= t for all k, sum(h)=1, h>=0
    
    # Build the constraint matrix for autoconvolution peaks
    # (h*h)_k = sum_i h_i * h_{i+k} for k in [-(N-1), N-1]
    # This is quadratic, not linear! So LP doesn't directly apply.
    # 
    # Alternative: Use the fact that for nonnegative h, the peak of autoconvolution
    # is at least sum(h)^2 / N (by Cauchy-Schwarz or similar bounds)
    # 
    # Let's try a different approach: minimize the L2 norm of autoconvolution
    # or use a smooth approximation.
    
    # Actually, let me reconsider. The ratio is:
    # R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # With ||h||_1 = 1: R = ||h||_2^2 / ||h*h||_inf
    # 
    # To maximize R, we want to maximize ||h||_2^2 and minimize ||h*h||_inf
    # 
    # For a given support pattern, ||h||_2^2 is maximized by concentrating mass
    # while ||h*h||_inf is minimized by spreading it out.
    # 
    # Let's use the smooth surrogate approach but with better initialization
    # and more iterations.
    
    beta = 50.0

    def smooth_neg_obj(h):
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-12:
            return 0.0
        hc = np.convolve(h, h, mode='full')
        if np.max(hc) < 1e-12:
            return 0.0
        scaled = beta * hc
        max_scaled = np.max(scaled)
        soft_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / beta
        return -(l2_sq / soft_max)

    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
    ]
    bounds = [(0.0, 1.0)] * N

    # Phase 1: Optimize with smoothed objective from Sidon init
    if time.time() < deadline - 30:
        try:
            res = minimize(smooth_neg_obj, best_h, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                h_candidate = np.clip(res.x, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # Phase 2: Try different initializations - sparse patterns
    if time.time() < deadline - 15:
        # Try single peak
        for init_type in ['single', 'two_peaks', 'alternating']:
            if time.time() > deadline - 10:
                break
            h_init = np.zeros(N)
            if init_type == 'single':
                h_init[N//2] = 1.0
            elif init_type == 'two_peaks':
                h_init[N//4] = 0.5
                h_init[3*N//4] = 0.5
            elif init_type == 'alternating':
                h_init[::2] = 1.0 / (N//2 + 1)
            
            s = np.sum(h_init)
            if s > 0:
                h_init /= s
            
            try:
                res = minimize(smooth_neg_obj, h_init, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
                if res.success:
                    h_candidate = np.clip(res.x, 0, None)
                    s = np.sum(h_candidate)
                    if s > 0:
                        h_candidate /= s
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
            except Exception:
                pass

    # Phase 3: Polish with exact max objective
    if time.time() < deadline - 5:
        def exact_neg_obj(h):
            l1 = np.sum(h)
            if l1 < 1e-12:
                return 0.0
            l2_sq = np.sum(h**2)
            hc = np.convolve(h, h, mode='full')
            mc = np.max(hc)
            if mc < 1e-12:
                return 0.0
            return -(l2_sq / mc)

        try:
            res = minimize(exact_neg_obj, best_h, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 400, 'ftol': 1e-12})
            if res.success:
                h_candidate = np.clip(res.x, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)
