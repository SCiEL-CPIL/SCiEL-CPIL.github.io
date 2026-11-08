import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    hh = np.convolve(h, h, mode='full')
    hh_inf = np.max(np.abs(hh))
    if hh_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * hh_inf)

def _negative_ratio(h):
    return -_ratio(h)

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Smooth surrogate for max to enable gradient methods
    def _smooth_ratio(h, beta=20.0):
        h = np.maximum(h, 0.0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        l2_sq = np.sum(h ** 2)
        hh = np.convolve(h, h, mode='full')
        hh_pos = np.maximum(hh, 0.0)
        # Softmax approximation of max
        m = np.max(hh_pos)
        if m < 1e-15:
            return 0.0
        hh_norm = hh_pos / m
        soft_max = m * np.log(np.sum(np.exp(beta * (hh_norm - 1.0)))) / beta
        if soft_max < 1e-15:
            return 0.0
        return l2_sq / (l1 * soft_max)
    
    def _neg_smooth_ratio(h, beta=20.0):
        return -_smooth_ratio(h, beta)
    
    # Strategy 1: Optimize over reduced support patterns
    # Try geometric sequences, triangular, etc.
    def _geometric_seq(n, r):
        h = np.zeros(N)
        vals = r ** np.arange(n)
        h[:n] = vals
        return h
    
    # Try various structured patterns
    patterns = []
    # Geometric sequences of various lengths and ratios
    for n in range(2, 15):
        for r in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
            patterns.append(_geometric_seq(n, r))
    # Triangular
    for n in range(3, 15):
        h = np.zeros(N)
        h[:n] = np.minimum(np.arange(n), n - 1 - np.arange(n)) + 1
        patterns.append(h)
    # Bimodal
    for n1 in range(1, 10):
        for n2 in range(1, 10):
            for gap in range(0, 5):
                h = np.zeros(N)
                if n1 + gap + n2 <= N:
                    h[:n1] = 1.0
                    h[n1+gap:n1+gap+n2] = 1.0
                    patterns.append(h)
    
    for p in patterns:
        if time.time() > deadline - 10:
            break
        s = _ratio(p)
        if s > best_score:
            best_score = s
            best_h = p.copy()
    
    # Strategy 2: Local optimization with smooth surrogate (Nelder-Mead on active support)
    # Identify active support from best
    active = np.where(best_h > 1e-8)[0]
    if len(active) == 0:
        active = np.arange(N)
    
    # Optimize weights on active support using smooth ratio
    def _opt_weights(x):
        h = np.zeros(N)
        h[active] = np.maximum(x, 0.0)
        return _neg_smooth_ratio(h, beta=30.0)
    
    if time.time() < deadline - 20:
        try:
            x0 = best_h[active]
            if np.max(x0) < 1e-10:
                x0 = np.ones(len(active))
            res = minimize(_opt_weights, x0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            h_opt = np.zeros(N)
            h_opt[active] = np.maximum(res.x, 0.0)
            s_opt = _ratio(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
        except Exception:
            pass
    
    # Strategy 3: L-BFGS-B with smooth surrogate from multiple restarts
    rng = np.random.default_rng(123)
    for restart in range(100):
        if time.time() > deadline - 15:
            break
        # Random sparse initialization
        k = int(rng.integers(2, min(12, N)))
        idx = np.sort(rng.choice(N, size=k, replace=False))
        h = np.zeros(N)
        h[idx] = rng.uniform(0.5, 3.0, size=k)
        
        # Quick smooth optimization
        def _neg_smooth(h_flat, beta=25.0):
            return _neg_smooth_ratio(h_flat, beta)
        
        bounds = [(0.0, None)] * N
        try:
            res = minimize(_neg_smooth, h, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            h_opt = np.maximum(res.x, 0.0)
            s_opt = _ratio(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Strategy 4: Final polish with exact ratio using L-BFGS-B
    if time.time() < deadline - 8:
        try:
            bounds = [(0.0, None)] * N
            res = minimize(_negative_ratio, best_h, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            h_opt = np.maximum(res.x, 0.0)
            s_opt = _ratio(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
        except Exception:
            pass
    
    # Strategy 5: Try expanding support - add small entries near active ones
    if time.time() < deadline - 5:
        active = np.where(best_h > 1e-6)[0]
        candidates = set(active.tolist())
        for a in active:
            for d in [-2, -1, 1, 2]:
                if 0 <= a + d < N:
                    candidates.add(a + d)
        candidates = np.array(sorted(candidates))
        
        def _opt_expanded(x):
            h = np.zeros(N)
            h[candidates] = np.maximum(x, 0.0)
            return _neg_smooth_ratio(h, beta=30.0)
        
        x0 = np.zeros(len(candidates))
        x0[np.isin(candidates, active)] = best_h[active]
        try:
            res = minimize(_opt_expanded, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            h_opt = np.zeros(N)
            h_opt[candidates] = np.maximum(res.x, 0.0)
            s_opt = _ratio(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
        except Exception:
            pass
    
    return list(best_h)
# EVOLVE-BLOCK-END
