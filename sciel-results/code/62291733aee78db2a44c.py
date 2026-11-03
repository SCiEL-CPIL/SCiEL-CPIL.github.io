import numpy as np
import time
from scipy.optimize import minimize

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

    # Smoothed objective using log-sum-exp to approximate max
    # This makes the optimization landscape smoother
    def smooth_neg_obj(h, beta):
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-12:
            return 0.0
        hc = np.convolve(h, h, mode='full')
        if np.max(hc) < 1e-12:
            return 0.0
        # Log-sum-exp based soft max
        scaled = beta * hc
        max_scaled = np.max(scaled)
        soft_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / beta
        return -(l2_sq / soft_max)

    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
    ]
    bounds = [(0.0, 1.0)] * N

    # Phase 1: Optimize with smoothed objective, using multiple beta values
    betas = [50.0, 30.0, 20.0]
    for beta in betas:
        if time.time() >= deadline - 30:
            break
        try:
            res = minimize(lambda h: smooth_neg_obj(h, beta), best_h, method='SLSQP', 
                          bounds=bounds, constraints=constraints, 
                          options={'maxiter': 300, 'ftol': 1e-10})
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

    # Phase 2: Polish with exact max objective
    if time.time() < deadline - 10:
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
                          constraints=constraints, options={'maxiter': 500, 'ftol': 1e-14})
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

    # Phase 3: High-precision polish with even tighter tolerance
    if time.time() < deadline - 2:
        def exact_neg_obj_tight(h):
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
            res = minimize(exact_neg_obj_tight, best_h, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 800, 'ftol': 1e-16})
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
