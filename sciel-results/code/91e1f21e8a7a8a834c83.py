import numpy as np
import time
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # 99
W = 1.0 / (M + 1)  # 1/100
TIME_LIMIT = 150.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # The objective is to maximize value = l2 / (norm_1 * norm_inf)
    # g = np.convolve(h, h), m = 2*N - 1
    # y = [0, g[0], ..., g[m-1], 0]
    # w = 1/(m+1)
    # l2 = sum_{i=0}^{m} (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # norm_1 = sum(g)/(m+1)
    # norm_inf = max(g)
    # value = l2 / (norm_1 * norm_inf)
    
    # Scale-invariant: fix sum(h) = 1.
    # Use smooth surrogate for max(g) via log-sum-exp to enable gradient-based optimization.
    
    def compute_value(h):
        h = np.asarray(h, dtype=float).ravel()
        if h.size != N:
            h = h[:N]
            if h.size < N:
                h = np.pad(h, (0, N - h.size), 'constant')
        if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
            return 0.0
        h = np.maximum(0.0, h)
        g = np.convolve(h, h)
        m = len(g)
        y = np.zeros(m + 2)
        y[1:-1] = g
        a = y[:m + 1]
        b = y[1:m + 2]
        l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
        norm_1 = np.sum(g) / (m + 1)
        norm_inf = np.max(g)
        if norm_1 <= 0 or norm_inf <= 0:
            return 0.0
        return l2 / (norm_1 * norm_inf)

    def _value_neg(h):
        v = compute_value(h)
        return -v

    def _smooth_max(g, t=50.0):
        # Smooth approximation of max(g) using log-sum-exp
        g_max = np.max(g)
        if g_max <= 0:
            return 0.0
        shifted = g - g_max
        return g_max + np.log(np.sum(np.exp(t * shifted))) / t

    def _value_neg_smooth(h, t=50.0):
        h = np.asarray(h, dtype=float).ravel()
        if h.size != N:
            h = h[:N]
            if h.size < N:
                h = np.pad(h, (0, N - h.size), 'constant')
        if not np.all(np.isfinite(h)):
            return 0.0
        h = np.maximum(0.0, h)
        g = np.convolve(h, h)
        m = len(g)
        y = np.zeros(m + 2)
        y[1:-1] = g
        a = y[:m + 1]
        b = y[1:m + 2]
        l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
        norm_1 = np.sum(g) / (m + 1)
        norm_inf_smooth = _smooth_max(g, t)
        if norm_1 <= 0 or norm_inf_smooth <= 0:
            return 0.0
        return -(l2 / (norm_1 * norm_inf_smooth))

    # Use softmax parameterization for unconstrained optimization
    def softmax_obj(x, t=50.0):
        h = np.exp(x - np.max(x))
        h = h / np.sum(h)
        return _value_neg_smooth(h, t)

    # Generate diverse starting points
    def _generate_starts():
        starts = []
        # Uniform
        starts.append(np.zeros(N))
        # Plateau-like: concentrated in center
        for center_frac in [0.3, 0.5, 0.7]:
            h_start = np.zeros(N)
            center = int(center_frac * N)
            width = max(1, N // 10)
            for i in range(N):
                dist = abs(i - center)
                h_start[i] = 1.0 / (1.0 + dist**2)
            starts.append(np.log(h_start + 1e-10))
        # Gaussian-like
        for sigma in [0.1, 0.2, 0.3]:
            x_idx = np.arange(N)
            h_start = np.exp(-((x_idx - N/2)**2) / (2 * sigma**2 * N**2))
            starts.append(np.log(h_start + 1e-10))
        # Random Dirichlet
        rng = np.random.RandomState(42)
        for i in range(8):
            h_start = rng.dirichlet(np.ones(N))
            starts.append(np.log(h_start + 1e-10))
        return starts

    best_val = 0.0
    best_h = np.ones(N) / N

    # Seed value
    h_seed = np.ones(N) / N
    best_val = compute_value(h_seed)
    best_h = h_seed.copy()

    starts = _generate_starts()

    # Phase 1: Smooth surrogate optimization with L-BFGS-B
    for x0 in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(softmax_obj, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                h_res = np.exp(res.x - np.max(res.x))
                h_res = h_res / np.sum(h_res)
                v_res = compute_value(h_res)
                if v_res > best_val:
                    best_val = v_res
                    best_h = h_res
        except Exception:
            pass

    # Phase 2: Refine with exact objective using SLSQP
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(N)]

    # Use best_h from Phase 1 as primary start
    start_list = [best_h.copy()]
    # Add a few more diverse starts
    for x0 in starts[:5]:
        h0 = np.exp(x0 - np.max(x0))
        h0 = h0 / np.sum(h0)
        start_list.append(h0)

    for h0 in start_list:
        if time.time() > deadline:
            break
        try:
            res = minimize(_value_neg, h0, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                v_res = -res.fun
                if v_res > best_val:
                    best_val = v_res
                    best_h = res.x
        except Exception:
            pass

    # Phase 3: High-precision polish
    if time.time() < deadline - 10:
        try:
            res = minimize(_value_neg, best_h, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                v_res = -res.fun
                if v_res > best_val:
                    best_val = v_res
                    best_h = res.x
        except Exception:
            pass

    # Ensure nonnegativity and finite
    best_h = np.maximum(0.0, best_h)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N) / N

    return list(best_h)
# EVOLVE-BLOCK-END
