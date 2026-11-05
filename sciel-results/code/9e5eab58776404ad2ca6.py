"""Optimized program for third autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_value(h):
    """Compute the objective value for a given h."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s ** 2))

def _soft_max(x, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def _objective(x):
    """Objective function using soft-max for differentiability."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    sm = _soft_max(conv, beta=50.0)
    return abs(2 * N * sm / (s ** 2))

def _objective_hard(x):
    """Exact objective with hard max."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s ** 2))

def construct():
    """Construct step heights to minimize the autocorrelation inequality constant."""
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # Should be 2.0
    
    def try_candidate(h):
        nonlocal best_h, best_val
        h = np.asarray(h, dtype=np.float64)
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-15:
            return
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Phase 1: Try structured initializations
    # Pattern: mostly negative with a few positive to make sum nonzero
    # Key idea: make most convolution entries very negative, center is sum(h^2)
    # but if sum(h) is large relative to sum(h^2), value can be small
    
    # Try: h with one large positive and many small negatives
    for frac_pos in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
        n_pos = max(1, int(frac_pos * N))
        for pos_val in [1.0, 2.0, 5.0, 10.0]:
            for neg_val in [-0.1, -0.5, -1.0, -2.0]:
                h = np.full(N, neg_val)
                h[:n_pos] = pos_val
                try_candidate(h)
    
    # Try: h = a * ones + b * random_signs
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 60:
            break
        a = np.random.uniform(0.5, 5.0)
        b = np.random.uniform(-3.0, 3.0)
        signs = np.random.choice([-1, 1], N)
        h = a + b * signs
        try_candidate(h)
    
    # Phase 2: Gradient-based optimization with soft-max
    def run_optimization(start_h, max_iter=5000):
        nonlocal best_h, best_val
        start_h = np.asarray(start_h, dtype=np.float64)
        s = np.sum(start_h)
        if abs(s) < 1e-10:
            start_h[0] += 1.0
        
        def obj(x):
            h = x
            s_val = np.sum(h)
            if abs(s_val) < 1e-10:
                return 1e10
            conv = np.convolve(h, h, mode='full')
            sm = _soft_max(conv, beta=50.0)
            return 2 * N * sm / (s_val ** 2)
        
        try:
            res = minimize(obj, start_h, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-10})
            if res.fun < best_val:
                # Verify with hard max
                hard_val = _objective_hard(res.x)
                if hard_val < best_val:
                    best_val = hard_val
                    best_h = res.x.copy()
                elif res.fun < best_val:
                    # Soft max is lower but hard might be higher, still try
                    try_candidate(res.x)
        except Exception:
            pass
    
    # Run optimization from multiple starting points
    starting_points = [
        np.ones(N),
        np.ones(N) * 0.5,
        np.linspace(1, -1, N),
        np.cos(2 * np.pi * np.arange(N) / N),
        np.sin(2 * np.pi * np.arange(N) / N),
    ]
    
    np.random.seed(123)
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        starting_points.append(np.random.randn(N))
    
    for sp in starting_points:
        if time.time() > deadline - 10:
            break
        run_optimization(sp, max_iter=2000)
    
    # Phase 3: Local refinement with Nelder-Mead on best found
    if time.time() < deadline - 5:
        def obj_nm(x):
            h = x
            s_val = np.sum(h)
            if abs(s_val) < 1e-10:
                return 1e10
            conv = np.convolve(h, h, mode='full')
            return abs(2 * N * np.max(conv) / (s_val ** 2))
        
        try:
            res = minimize(obj_nm, best_h, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_val:
                hard_val = _objective_hard(res.x)
                if hard_val < best_val:
                    best_val = hard_val
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Try more aggressive structured patterns
    if time.time() < deadline - 5:
        # Two-block with optimized ratio
        for k in range(1, N, 10):
            if time.time() > deadline - 3:
                break
            for ratio in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                h = np.zeros(N)
                h[:k] = ratio
                h[k:] = -1.0
                try_candidate(h)
        
        # Three-block
        for k1 in range(1, N//3, 20):
            if time.time() > deadline - 2:
                break
            for k2 in range(k1+1, 2*N//3, 20):
                if time.time() > deadline - 1:
                    break
                for v1 in [1.0, 2.0]:
                    for v2 in [-0.5, -1.0, -2.0]:
                        for v3 in [0.5, 1.0]:
                            h = np.zeros(N)
                            h[:k1] = v1
                            h[k1:k2] = v2
                            h[k2:] = v3
                            try_candidate(h)
    
    return best_h
