import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the score: L2^2 / (L1 * Sup(autoconv))
    For step heights h on intervals of width DELTA:
    L1 = DELTA * sum(h)
    L2^2 = DELTA * sum(h^2)
    autoconv_max = DELTA^2 * max(conv(h, h))
    
    Ratio = (DELTA * sum(h^2)) / (DELTA * sum(h) * DELTA^2 * max_conv)
          = sum(h^2) / (DELTA^2 * sum(h) * max_conv)
    
    Since DELTA is constant, maximizing the raw score is equivalent to maximizing
    J(h) = sum(h^2) / (sum(h) * max_conv)
    
    If we normalize sum(h) = 1, we maximize sum(h^2) / max_conv.
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
    # The raw score includes the 1/DELTA^2 factor
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # 1. Find a large Sidon set in {0, ..., N-1}
    # A Sidon set (B_2 set) has distinct pair sums i+j (i <= j).
    # We want the largest such set to minimize convolution collisions.
    
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
        # Greedy approach
        S = [0]
        for i in range(1, N):
            candidate = S + [i]
            if is_sidon(candidate):
                S = candidate
        return S

    sidon_set = find_sidon_set(N)
    K = len(sidon_set)
    
    # 2. Initialize best_h with uniform weights on Sidon set
    best_h = np.zeros(N)
    best_h[sidon_set] = 1.0 / K
    best_score = _score_raw(best_h)
    
    # 3. Optimize weights on the fixed Sidon support
    # We want to maximize J(h) = sum(h^2) / max_conv subject to sum(h)=1, h>=0.
    # Since the support is Sidon, max_conv = max_{i,j in S} h_i h_j.
    # If we order h such that h_1 >= h_2 >= ... >= h_K, then max_conv = h_1 h_2 (if K>=2) or h_1^2 (if K=1).
    # Actually, max_conv is the max of all products.
    # For skewed weights, h_1 h_2 is likely the max cross term, and h_1^2 is the max self term.
    # So max_conv = max(h_1^2, h_1 h_2).
    # If h_1 > h_2, h_1^2 > h_1 h_2. So max_conv = h_1^2.
    # Then J = sum(h^2) / h_1^2 = 1 + sum_{i>1} h_i^2 / h_1^2.
    # To maximize this, we want sum_{i>1} h_i^2 to be large relative to h_1^2.
    # But h_1 is the largest, so h_1^2 dominates.
    # Wait, if max_conv = h_1^2, then J is just 1 + small.
    # But if the set is NOT Sidon, max_conv could be larger.
    # With Sidon, we avoid collisions.
    
    # Let's try to make h_1 and h_2 close to each other to increase h_1 h_2?
    # No, we want to MINIMIZE max_conv.
    # If max_conv is determined by h_1^2, we want h_1 to be small.
    # But h_1 is the largest.
    # If we make all h_i equal, h_1 = 1/K.
    # max_conv = 1/K^2.
    # J = (1/K) / (1/K^2) = K.
    
    # If we skew: h_1=0.5, h_2=0.5, rest 0.
    # max_conv = max(0.25, 0.25, 0.25) = 0.25.
    # sum_sq = 0.5.
    # J = 2.
    # If K=2, uniform gives J=2. Skewed gives J=2.
    
    # If K=3, uniform: h=[1/3, 1/3, 1/3].
    # max_conv = 1/9.
    # sum_sq = 3/9 = 1/3.
    # J = 3.
    
    # Skewed: h=[0.5, 0.25, 0.25].
    # max_conv = max(0.25, 0.125, 0.0625) = 0.25.
    # sum_sq = 0.25 + 0.0625 + 0.0625 = 0.375.
    # J = 0.375 / 0.25 = 1.5.
    # So uniform is better for Sidon sets!
    
    # So the optimal strategy for a Sidon set is likely UNIFORM weights.
    # Score should be K.
    
    # Let's verify with SLSQP just in case there's a better weight distribution.
    # Objective: minimize -sum(h^2) / max(conv(h,h))
    
    def neg_obj(h):
        l1 = np.sum(h)
        if l1 < 1e-12: return 0
        l2_sq = np.sum(h**2)
        hc = np.convolve(h, h, mode='full')
        mc = np.max(hc)
        if mc < 1e-12: return 0
        return -(l2_sq / mc)
        
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    # Try optimizing from the uniform Sidon init
    if time.time() < deadline - 10:
        try:
            res = minimize(neg_obj, best_h, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                h_candidate = res.x
                h_candidate = np.clip(h_candidate, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
            
    # Also try a few random Sidon-like sets to see if a larger K is possible or better structure
    # The greedy Sidon set might not be maximal.
    # But for N=50, K~7 is good.
    
    return list(best_h)
