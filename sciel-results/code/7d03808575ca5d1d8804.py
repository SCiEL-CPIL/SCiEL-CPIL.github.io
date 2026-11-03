import numpy as np
import time
import sys
from scipy.optimize import minimize, LinearConstraint

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _score_normalized(h):
    """Score with L1 normalization: L2^2 / max(conv(h,h)) with sum(h)=1."""
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if s == 0:
        return 0.0
    h_norm = h / s
    l2_sq = np.sum(h_norm**2)
    h_conv = np.convolve(h_norm, h_norm, mode='full')
    max_conv = np.max(h_conv)
    if max_conv == 0:
        return 0.0
    return l2_sq / max_conv

def _softmax_conv(h, temperature=1.0):
    """Smooth surrogate for max(conv(h,h)) using log-sum-exp."""
    h_conv = np.convolve(h, h, mode='full')
    # Use scaled log-sum-exp for numerical stability
    max_val = np.max(h_conv)
    if max_val <= 0:
        return 0.0
    # temperature controls smoothness; smaller = closer to max
    log_sum = np.log(np.sum(np.exp(h_conv / temperature))) * temperature
    return log_sum

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    best_h = np.ones(N)
    best_score = _score_raw(best_h)

    x_idx = np.arange(N)

    # 1. Uniform
    h_init = np.ones(N)
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()

    # 2. Gaussian
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()

    # 3. Triangular
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()

    # 4. Random sparse initialization
    for _ in range(100):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        h_rand += np.random.rand(N) * 0.1
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # Optimization using SLSQP on log-scale variables with smooth surrogate
    # h_i = exp(x_i)
    # Use log-sum-exp surrogate for max(conv) to get smooth objective

    def neg_score_smooth(x, temp=1.0):
        h = np.exp(x)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        # Soft max of convolution
        h_conv = np.convolve(h, h, mode='full')
        max_val = np.max(h_conv)
        if max_val <= 0:
            return 0.0
        # Log-sum-exp approximation to max
        # For stability: max + log(sum(exp((conv - max)/temp))) * temp
        shifted = (h_conv - max_val) / temp
        log_sum = max_val + temp * np.log(np.sum(np.exp(shifted)))
        if log_sum <= 0:
            return 0.0
        score = l2_sq / (l1 * log_sum)
        if score <= 0:
            return 0.0
        return -score

    # Start from best_h
    h_opt_init = best_h.copy()
    h_opt_init[h_opt_init < 1e-8] = 1e-8
    x0 = np.log(h_opt_init)

    # Try multiple temperatures for the soft-max surrogate
    temperatures = [1.0, 0.1, 0.01, 0.001]

    for temp in temperatures:
        if time.time() > deadline:
            break

        def neg_score_temp(x, t=temp):
            return neg_score_smooth(x, t)

        try:
            res = minimize(neg_score_temp, x0, method='BFGS',
                          options={'maxiter': 500, 'gtol': 1e-8})
            if res.success:
                h_candidate = np.exp(res.x)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
                    x0 = res.x.copy()
        except Exception:
            pass

    # Also try SLSQP on normalized h directly
    def neg_obj_h(h):
        l1 = np.sum(h)
        if l1 == 0: return 0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0: return 0
        return -(l2_sq / max_conv)

    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h}
    ]
    bounds = [(0.0, 1.0)] * N

    # Generate diverse initializations
    inits = []
    inits.append(best_h / np.sum(best_h))
    inits.append(np.ones(N) / N)

    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N/2, 5):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)

    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N, 5):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)

    for _ in range(20):
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)

    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)

    # Optimize from each init
    for h_init in inits:
        if time.time() > deadline: break
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)

        try:
            res = minimize(neg_obj_h, h_init, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Coordinate descent polish: iterate over each coordinate, optimize it while fixing others
    if time.time() < deadline - 20:
        h_cd = best_h / np.sum(best_h)
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                # Optimize h[i] while fixing all other coordinates
                # With sum constraint: h[i] = 1 - sum(h[j] for j != i)
                # But we also need h[i] >= 0, so h[i] in [0, 1 - sum_others]
                others_sum = np.sum(h_cd) - h_cd[i]
                if others_sum >= 1.0:
                    continue  # Can't increase h[i]
                # Try a grid of values for h[i]
                # Actually, let's just do a local search
                h_old = h_cd[i]
                # Perturb h[i] and renormalize
                for delta in [-0.01, -0.005, -0.001, 0.001, 0.005, 0.01]:
                    h_try = h_cd.copy()
                    h_try[i] = max(0, h_old + delta)
                    s_try = np.sum(h_try)
                    if s_try == 0:
                        continue
                    h_try /= s_try
                    s_score = _score_raw(h_try)
                    if s_score > best_score:
                        best_score = s_score
                        best_h = h_try.copy()
                        h_cd = h_try.copy()
                        improved = True
                        break
            if not improved:
                break

    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass

    return list(best_h)

# EVOLVE-BLOCK-END
