import numpy as np
import time
from scipy.optimize import minimize

N = 50
w = 1.0 / 100.0  # spacing on [-0.5, 0.5] with m+2=102 points

def compute_value(h):
    """Exact evaluation matching the evaluator formula."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.clip(h, 0.0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def compute_value_fast(h):
    """Fast evaluation using algebraic simplification."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.clip(h, 0.0, None)
    g = np.convolve(h, h)
    m = len(g)
    g_sq_sum = np.sum(g**2)
    cross_sum = np.sum(g[:-1] * g[1:]) if m > 1 else 0.0
    l2 = (w / 3.0) * (2.0 * g_sq_sum + cross_sum)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def neg_value_smooth(h, beta=50.0):
    """Negative of smoothed value with soft-max for norm_inf."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return 1e10
    h = np.clip(h, 0.0, None)
    g = np.convolve(h, h)
    m = len(g)
    g_sq_sum = np.sum(g**2)
    cross_sum = np.sum(g[:-1] * g[1:]) if m > 1 else 0.0
    l2 = (w / 3.0) * (2.0 * g_sq_sum + cross_sum)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    
    # Soft-max surrogate
    g_max = np.max(g)
    if g_max > 0:
        log_sum_exp = g_max + np.log(np.sum(np.exp(g - g_max)))
        soft_max = log_sum_exp / beta
    else:
        soft_max = 0.0
    
    if norm_1 * soft_max < 1e-30:
        return 1e10
    return -l2 / (norm_1 * soft_max)

def neg_value_exact(h):
    """Negative of exact value for final polish."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return 1e10
    h = np.clip(h, 0.0, None)
    val = compute_value(h)
    if val <= 0:
        return 1e10
    return -val

def _hump_init():
    x = np.linspace(-1, 1, N)
    h = np.exp(-2 * x**2)
    h = h / np.max(h)
    return h

def _two_peak_init():
    x = np.linspace(-1, 1, N)
    h = np.exp(-4 * (x - 0.5)**2) + np.exp(-4 * (x + 0.5)**2)
    h = h / np.max(h)
    return h

def _ramp_init():
    return np.linspace(0.1, 1.0, N)

def _bump_init(width=0.3):
    x = np.linspace(-1, 1, N)
    h = np.exp(-1.5 * (x / width)**2)
    h = h / np.max(h)
    return h

def _triangle_init():
    x = np.linspace(-1, 1, N)
    h = np.maximum(0, 1 - np.abs(x))
    h = h / np.max(h)
    return h

def _uniform_init():
    return np.ones(N)

def _random_smooth(rng):
    x = rng.randn(N)
    kernel = np.ones(9) / 9.0
    x = np.convolve(x, kernel, mode='same')
    x = x - x.mean()
    return np.maximum(0.1 + x, 0.0)

def _optimize_from_start(h0, deadline):
    """Multi-stage optimization: smooth surrogates then exact polish."""
    if time.time() > deadline:
        return h0, compute_value(h0)
    
    best_h = np.array(h0, dtype=np.float64)
    best_val = compute_value(best_h)
    
    # Stage 1: Coarse smooth optimization with small beta
    betas = [5.0, 15.0, 40.0, 100.0, 300.0]
    for beta in betas:
        if time.time() > deadline:
            break
        
        def _neg_smooth(h):
            return neg_value_smooth(h, beta=beta)
        
        try:
            res = minimize(
                _neg_smooth,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            h_opt = res.x
            val_opt = compute_value(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Stage 2: Near-exact polish with very large beta
    if time.time() < deadline - 2.0:
        def _neg_near_exact(h):
            return neg_value_smooth(h, beta=1000.0)
        
        try:
            res = minimize(
                _neg_near_exact,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 30}
            )
            h_opt = res.x
            val_opt = compute_value(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Stage 3: Final exact polish
    if time.time() < deadline - 0.5:
        try:
            res = minimize(
                neg_value_exact,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 20}
            )
            h_opt = res.x
            val_opt = compute_value(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    return best_h, best_val

def _coordinate_descent(h, deadline, max_rounds=50):
    """Coordinate descent with adaptive step sizes."""
    h = h.copy()
    best_val = compute_value(h)
    n = len(h)
    
    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001, -0.1, -0.05, -0.01, -0.005, -0.001]
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            orig = h[i]
            for delta in step_sizes:
                cand = h.copy()
                cand[i] = max(0.0, orig + delta)
                v = compute_value(cand)
                if v > best_val + 1e-14:
                    best_val = v
                    h = cand
                    improved = True
                    break
        
        if not improved:
            # Try Nelder-Mead on random subset
            subset_size = min(8, n)
            subset = np.random.choice(n, size=subset_size, replace=False)
            sub_h = h[subset]
            try:
                def sub_obj(x):
                    full = h.copy()
                    full[subset] = np.maximum(x, 0.0)
                    return -compute_value(full)
                
                res = minimize(
                    sub_obj,
                    sub_h,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                h_cand = h.copy()
                h_cand[subset] = np.maximum(res.x, 0.0)
                v_cand = compute_value(h_cand)
                if v_cand > best_val:
                    best_val = v_cand
                    h = h_cand.copy()
                    improved = True
            except Exception:
                pass
        
        if not improved and rnd > 10:
            break
    
    return h, best_val

def construct():
    deadline = time.time() + 130.0  # Safety margin before 150s kill
    rng = np.random.RandomState(42)
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Phase 1: Try various initializations with multi-stage optimization
    inits = [
        _uniform_init,
        _hump_init,
        _two_peak_init,
        _ramp_init,
        _bump_init,
        _triangle_init,
    ]
    
    for init_func in inits:
        if time.time() > deadline - 70.0:
            break
        h0 = init_func()
        h_opt, val_opt = _optimize_from_start(h0, deadline - 60.0)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Phase 2: More diverse random starts
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline - 45.0:
            break
        
        if i == 0:
            x = _random_smooth(rng)
        elif i == 1:
            x = np.sin(np.pi * np.arange(N) / N) * 0.5 + 0.5
        elif i == 2:
            x = np.cos(2*np.pi*np.arange(N)/N) * 0.4 + 0.6
        elif i == 3:
            x = (1 - 2*np.abs(np.arange(N)/N - 0.5))**2 + 0.1
        elif i == 4:
            x = np.exp(-((np.arange(N)-N//2)**2)/20.0) + 0.1
        elif i == 5:
            x = np.where(np.arange(N) < N//2, 0.3, 0.7) + 0.05*rng.randn(N)
        elif i == 6:
            x = np.exp(-((np.arange(N)-N//4)**2)/8.0) + np.exp(-((np.arange(N)-3*N//4)**2)/8.0) + 0.1
        elif i == 7:
            x = np.linspace(0.2, 0.8, N)
        elif i == 8:
            x = 0.3*np.cos(4*np.pi*np.arange(N)/N) + 0.5
        else:
            x = 0.3*np.sin(3*np.pi*np.arange(N)/N) + 0.5
        
        x = np.maximum(x, 1e-6)
        h_opt, val_opt = _optimize_from_start(x, deadline - 35.0)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Phase 3: Coordinate descent refinement
    if time.time() < deadline - 20.0:
        best_h, best_val = _coordinate_descent(best_h, deadline - 15.0, max_rounds=30)
    
    # Phase 4: Final multi-stage polish
    if time.time() < deadline - 5.0:
        best_h, best_val = _optimize_from_start(best_h, deadline - 2.0)
    
    # Phase 5: Final coordinate descent
    if time.time() < deadline - 2.0:
        best_h, best_val = _coordinate_descent(best_h, deadline - 0.5, max_rounds=20)
    
    # Ensure nonnegativity and finiteness
    best_h = np.maximum(np.asarray(best_h, dtype=np.float64), 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)
