import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
w = 1.0 / 100.0  # spacing on [-0.5, 0.5] with m+2=102 points

def compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)  # 99
    y = np.concatenate(([0.0], g, [0.0]))  # length m+2 = 101
    sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def compute_value_fast(h):
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

def compute_value_precise(h):
    """High-precision evaluation using full formula with careful summation."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    # Use pairwise summation for better precision
    term1 = y[:m+1]**2
    term2 = y[:m+1] * y[1:m+2]
    term3 = y[1:m+2]**2
    sq = term1 + term2 + term3
    l2 = (w / 3.0) * np.sum(sq, dtype=np.float64)
    norm_1 = np.sum(np.abs(g), dtype=np.float64) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def neg_value(log_h):
    h = np.exp(log_h)
    return -compute_value_fast(h)

def neg_value_direct(h):
    return -compute_value_fast(h)

def neg_value_smooth(log_h):
    """Use a smoother objective that helps gradient methods."""
    h = np.exp(log_h)
    g = np.convolve(h, h)
    m = len(g)
    g_sq_sum = np.sum(g**2)
    cross_sum = np.sum(g[:-1] * g[1:]) if m > 1 else 0.0
    l2 = (w / 3.0) * (2.0 * g_sq_sum + cross_sum)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    # Use a smoothed max instead of hard max for better gradients
    norm_inf_smooth = np.logaddexp.reduce(np.log(np.abs(g) + 1e-10))
    if norm_inf_smooth == 0 or norm_1 == 0:
        return 0.0
    return -l2 / (norm_1 * norm_inf_smooth)

def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # should be 1.0

    rng = np.random.RandomState(42)

    def try_optimize_log(start_log_h, maxiter, method='L-BFGS-B'):
        nonlocal best_h, best_val
        t0 = time.time()
        try:
            if method == 'L-BFGS-B':
                res = minimize(
                    neg_value,
                    start_log_h,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-14, 'gtol': 1e-12, 'eps': 1e-8}
                )
            elif method == 'BFGS':
                res = minimize(
                    neg_value,
                    start_log_h,
                    method='BFGS',
                    options={'maxiter': maxiter, 'gtol': 1e-10}
                )
            else:
                res = minimize(
                    neg_value_smooth,
                    start_log_h,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-14, 'gtol': 1e-12, 'eps': 1e-8}
                )
            h_cand = np.exp(res.x)
            v_cand = compute_value_fast(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
        return time.time() - t0

    def try_optimize_direct(start_h, maxiter):
        nonlocal best_h, best_val
        t0 = time.time()
        try:
            res = minimize(
                neg_value_direct,
                start_h,
                method='L-BFGS-B',
                bounds=[(1e-10, 100.0)] * N,
                options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-14, 'gtol': 1e-12, 'eps': 1e-8}
            )
            h_cand = np.maximum(res.x, 0.0)
            v_cand = compute_value_fast(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
        return time.time() - t0

    def try_differential_evolution(bounds_log, maxiter):
        nonlocal best_h, best_val
        t0 = time.time()
        try:
            # DE in log-space
            def de_obj(log_h):
                h = np.exp(log_h)
                return -compute_value_fast(h)
            
            # Use bounds for log_h
            lb = bounds_log[0]
            ub = bounds_log[1]
            res = differential_evolution(
                de_obj,
                [(lb, ub)] * N,
                maxiter=maxiter,
                tol=1e-12,
                seed=42,
                polish=True,
                update_interval=10
            )
            h_cand = np.exp(res.x)
            v_cand = compute_value_fast(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
        return time.time() - t0

    def custom_local_search(start_h, n_iter=5000):
        """Custom local search with adaptive step size."""
        nonlocal best_h, best_val
        t0 = time.time()
        h = start_h.copy()
        current_val = compute_value_fast(h)
        best_local_val = current_val
        best_local_h = h.copy()
        step_size = 0.1
        
        for i in range(n_iter):
            if time.time() > deadline - 10.0:
                break
            
            # Generate random perturbation
            delta = rng.randn(N) * step_size
            h_new = np.maximum(h + delta, 0.0)
            v_new = compute_value_fast(h_new)
            
            if v_new > current_val:
                h = h_new
                current_val = v_new
                step_size *= 1.05
                if v_new > best_local_val:
                    best_local_val = v_new
                    best_local_h = h.copy()
            else:
                step_size *= 0.95
                # Sometimes try a larger jump
                if rng.random() < 0.1:
                    delta = rng.randn(N) * step_size * 5
                    h_new = np.maximum(h + delta, 0.0)
                    v_new = compute_value_fast(h_new)
                    if v_new > current_val:
                        h = h_new
                        current_val = v_new
                        step_size *= 1.1
                        if v_new > best_local_val:
                            best_local_val = v_new
                            best_local_h = h.copy()
        
        if best_local_val > best_val:
            best_val = best_local_val
            best_h = best_local_h.copy()
        return time.time() - t0

    # Start from uniform
    try_optimize_log(np.zeros(N), maxiter=8000)
    
    # Try smooth objective optimization
    if time.time() < deadline - 60.0:
        try_optimize_log(np.zeros(N), maxiter=5000, method='smooth')
    
    # Try a broad differential evolution search in log-space
    if time.time() < deadline - 60.0:
        try_differential_evolution((-5.0, 5.0), maxiter=100)
    
    # Additional targeted starts: emphasize different frequency components
    if time.time() < deadline - 50.0:
        # Triangle wave (emphasizes odd harmonics)
        x_tri = 1.0 - 2.0 * np.abs(2.0 * (np.arange(N) / N) - 1.0)
        x_tri = x_tri - x_tri.mean()
        x_tri = 0.5 * x_tri
        try_optimize_log(x_tri, maxiter=4000)
    
    if time.time() < deadline - 40.0:
        # Square-wave-like pattern
        x_sq = np.where(np.arange(N) % 2 == 0, 0.3, -0.3)
        x_sq = x_sq - x_sq.mean()
        try_optimize_log(x_sq, maxiter=4000)
    
    # Multi-start perturbations
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 40.0:
            break
        
        if i == 0:
            # Smooth random
            x = rng.randn(N)
            kernel = np.ones(7)/7.0
            x = np.convolve(x, kernel, mode='same')
            x = x - x.mean()
            x = 0.3 * x
        elif i == 1:
            x = np.sin(np.pi * np.arange(N) / N) * 0.5
        elif i == 2:
            x = np.cos(2*np.pi*np.arange(N)/N) * 0.4
        elif i == 3:
            x = 1 - 2*np.abs(np.arange(N)/N - 0.5)
            x = x - x.mean()
        elif i == 4:
            x = np.exp(-((np.arange(N)-N//2)**2)/20.0)
            x = x - x.mean()
        elif i == 5:
            x = np.where(np.arange(N) < N//2, -0.3, 0.3) + 0.1*rng.randn(N)
        elif i == 6:
            x = (1 - 2*np.abs(np.arange(N)/N - 0.5))**2
            x = x - x.mean()
        elif i == 7:
            x = np.exp(-((np.arange(N)-N//4)**2)/8.0) + np.exp(-((np.arange(N)-3*N//4)**2)/8.0)
            x = x - x.mean()
        elif i == 8:
            x = np.linspace(-0.4, 0.4, N)
        elif i == 9:
            x = 0.3*np.cos(4*np.pi*np.arange(N)/N)
        elif i == 10:
            x = np.exp(-((np.arange(N)-N//2)**2)/5.0)
            x = x - x.mean()
        elif i == 11:
            x = np.exp(-((np.arange(N)-N//2)**2)/100.0)
            x = x - x.mean()
        elif i == 12:
            x = 2*np.abs(np.arange(N)/N - 0.5)
            x = x - x.mean()
        elif i == 13:
            x = 0.3*np.sin(3*np.pi*np.arange(N)/N)
        elif i == 14:
            x = np.where((np.arange(N) > N//4) & (np.arange(N) < 3*N//4), 0.4, -0.2)
        elif i == 15:
            # High frequency oscillation
            x = 0.2*np.sin(10*np.pi*np.arange(N)/N)
        elif i == 16:
            # Asymmetric
            x = np.sin(np.pi * np.arange(N) / (2*N)) - 0.5
        elif i == 17:
            # Two peaks offset
            x = np.exp(-((np.arange(N)-N//3)**2)/15.0) + np.exp(-((np.arange(N)-2*N//3)**2)/15.0)
            x = x - x.mean()
        elif i == 18:
            # S-shaped
            x = np.tanh(np.linspace(-3, 3, N))
            x = x - x.mean()
        else:
            x = 0.3*rng.randn(N)
        
        x = np.maximum(x, -6.0)
        try_optimize_log(x, maxiter=6000)
    
    # Custom local search from best
    if time.time() < deadline - 30.0:
        custom_local_search(best_h.copy(), n_iter=3000)
    
    # Final polish rounds
    if time.time() < deadline - 20.0:
        try_optimize_direct(best_h.copy(), maxiter=10000)
    
    if time.time() < deadline - 15.0:
        try_optimize_log(np.log(np.maximum(best_h, 1e-10)), maxiter=12000)
    
    # One more DE from around best
    if time.time() < deadline - 20.0:
        log_best = np.log(np.maximum(best_h, 1e-10))
        bounds = [(max(log_best[i]-3, -6.0), min(log_best[i]+3, 6.0)) for i in range(N)]
        # Convert to simple bounds for DE
        lb = min(b[0] for b in bounds)
        ub = max(b[1] for b in bounds)
        try_differential_evolution((lb, ub), maxiter=80)
    
    # Final fine polish
    if time.time() < deadline - 5.0:
        try_optimize_log(np.log(np.maximum(best_h, 1e-10)), maxiter=5000)
    
    # High-precision verification and micro-polish
    if time.time() < deadline - 2.0:
        t0 = time.time()
        # Verify with precise computation
        val_precise = compute_value_precise(best_h)
        val_fast = compute_value_fast(best_h)
        
        # If there's a discrepancy, try a very tight local optimization
        if abs(val_precise - val_fast) > 1e-10 and time.time() - t0 < 1.0:
            try:
                # Tight local search in direct space with very small steps
                res = minimize(
                    neg_value_direct,
                    best_h,
                    method='L-BFGS-B',
                    bounds=[(0.0, 100.0)] * N,
                    options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-16, 'gtol': 1e-14, 'eps': 1e-10}
                )
                h_cand = np.maximum(res.x, 0.0)
                v_cand = compute_value_precise(h_cand)
                if v_cand > val_precise:
                    best_val = v_cand
                    best_h = h_cand.copy()
            except Exception:
                pass

    return list(np.asarray(best_h, dtype=float))
