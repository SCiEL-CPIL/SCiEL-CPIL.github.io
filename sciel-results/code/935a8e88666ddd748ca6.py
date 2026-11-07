import numpy as np
import time
from scipy.optimize import minimize

N = 50
w = 1.0 / 100.0  # spacing on [-0.5, 0.5] with m+2=102 points

def compute_value(h):
    """Compute the objective value for given step heights h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0
    
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft <<= 1
    
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:2*n-1]
    g = np.real(g)
    g = np.maximum(g, 0.0)
    
    m = len(g)
    
    g_sq = np.sum(g**2)
    cross = np.sum(g[:-1] * g[1:])
    l2 = (w / 3.0) * (2.0 * g_sq + cross)
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def neg_value_smooth(log_h, beta=100.0):
    """Smooth objective using log-sum-exp for the max term."""
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft <<= 1
    
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:2*n-1]
    g = np.real(g)
    g = np.maximum(g, 0.0)
    
    m = len(g)
    
    g_sq = np.sum(g**2)
    cross = np.sum(g[:-1] * g[1:])
    l2 = (w / 3.0) * (2.0 * g_sq + cross)
    
    sum_g = np.sum(g)
    norm_1 = sum_g / (m + 1)
    
    g_max_raw = np.max(g)
    if g_max_raw > 0:
        log_sum_exp = g_max_raw + np.log(np.sum(np.exp(beta * (g - g_max_raw))))
        norm_inf_smooth = (1.0 / beta) * log_sum_exp
    else:
        norm_inf_smooth = 0.0
    
    if norm_inf_smooth == 0 or norm_1 == 0:
        return 0.0
        
    return -l2 / (norm_1 * norm_inf_smooth)

def neg_value_direct(log_h):
    """Direct objective (exact max)."""
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value(h)

def construct():
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    rng = np.random.RandomState(42)
    
    def try_optimize_log(start_log_h, maxiter, use_smooth=True, beta=100.0):
        nonlocal best_h, best_val
        t0 = time.time()
        if use_smooth:
            obj = lambda x: neg_value_smooth(x, beta=beta)
        else:
            obj = neg_value_direct
        try:
            res = minimize(
                obj,
                start_log_h,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_cand = np.exp(res.x - np.mean(res.x))
            h_cand = np.maximum(h_cand, 0.0)
            v_cand = compute_value(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
        return time.time() - t0

    # Start from uniform
    try_optimize_log(np.zeros(N), maxiter=5000, use_smooth=True)

    # Multi-start perturbations
    n_starts = 25
    for i in range(n_starts):
        if time.time() > deadline - 35.0:
            break
        
        idx = np.arange(N)
        
        if i == 0:
            # Smooth random
            x = rng.randn(N)
            kernel = np.ones(5)/5.0
            x = np.convolve(x, kernel, mode='same')
            x = x - x.mean()
            x = 0.5 * x
        elif i == 1:
            # Sine
            x = np.sin(np.pi * idx / N) + 0.3
        elif i == 2:
            # Cosine
            x = np.cos(2*np.pi*idx/N) * 0.5 + 0.5
        elif i == 3:
            # Triangle
            x = 1 - 2*np.abs(idx/N - 0.5)
        elif i == 4:
            # Gaussian central
            x = np.exp(-((idx - N//2)**2)/30.0)
            x = x - x.mean() + 0.5
        elif i == 5:
            # Step
            x = np.where(idx < N//2, 0.2, 0.8) + 0.1*rng.randn(N)
        elif i == 6:
            # Triangle squared
            x = (1 - 2*np.abs(idx/N - 0.5))**2
        elif i == 7:
            # Two peaks
            x = np.exp(-((idx-N//4)**2)/10.0) + np.exp(-((idx-3*N//4)**2)/10.0)
            x = x - x.mean() + 0.5
        elif i == 8:
            # Ramp
            x = np.linspace(0.2, 0.8, N)
        elif i == 9:
            # Alternating
            x = 0.5 + 0.3*np.cos(4*np.pi*idx/N)
        elif i == 10:
            # Narrow central peak
            x = np.exp(-((idx-N//2)**2)/5.0)
            x = x - x.mean() + 0.5
        elif i == 11:
            # Broad central peak
            x = np.exp(-((idx-N//2)**2)/80.0)
            x = x - x.mean() + 0.5
        elif i == 12:
            # U-shape
            x = 2*np.abs(idx/N - 0.5)
            x = x - x.mean() + 0.5
        elif i == 13:
            # Sine higher freq
            x = 0.5 + 0.4*np.sin(3*np.pi*idx/N)
        elif i == 14:
            # Plateau
            x = np.where((idx > N//4) & (idx < 3*N//4), 0.9, 0.3)
        elif i == 15:
            # Hann window
            x = 0.5 * (1.0 + np.cos(2.0 * np.pi * idx / N))
        elif i == 16:
            # Double sine
            x = 0.5 + 0.3*np.sin(2*np.pi*idx/N) + 0.2*np.sin(4*np.pi*idx/N)
        elif i == 17:
            # Gaussian mixture
            x = 0.6*np.exp(-((idx-N//3)**2)/15.0) + 0.4*np.exp(-((idx-2*N//3)**2)/15.0)
            x = x - x.mean() + 0.5
        elif i == 18:
            # Flat-top with tapered edges
            x = np.ones(N)
            taper = 8
            for k in range(taper):
                frac = 0.5*(1.0 + np.cos(np.pi*(k+0.5)/taper))
                x[k] = frac
                x[N-1-k] = frac
        elif i == 19:
            # Cosine squared
            x = np.cos(np.pi * idx / N)**2
        elif i == 20:
            # Power law
            x = (idx/N)**0.5
            x = x - x.mean() + 0.5
        elif i == 21:
            # Inverse power
            x = 1.0 / (0.1 + idx/N)
            x = x - x.mean() + 0.5
        elif i == 22:
            # Three peaks
            x = np.exp(-((idx-N//6)**2)/8.0) + np.exp(-((idx-N//2)**2)/8.0) + np.exp(-((idx-5*N//6)**2)/8.0)
            x = x - x.mean() + 0.5
        elif i == 23:
            # Smooth random with different seed
            x = rng.randn(N)
            kernel = np.ones(7)/7.0
            x = np.convolve(x, kernel, mode='same')
            x = x - x.mean()
            x = 0.4 * x
        else:
            x = rng.randn(N) * 0.3
            kernel = np.ones(3)/3.0
            x = np.convolve(x, kernel, mode='same')
            x = x - x.mean()
        
        x = np.maximum(x, -5.0)
        try_optimize_log(x, maxiter=3000, use_smooth=True)

    # Final polish: direct optimization from best found
    if time.time() < deadline - 25.0:
        try_optimize_log(np.log(np.maximum(best_h, 1e-6)), maxiter=8000, use_smooth=False)

    # One more log-space polish
    if time.time() < deadline - 10.0:
        try_optimize_log(np.log(np.maximum(best_h, 1e-6)), maxiter=5000, use_smooth=True, beta=200.0)

    # High-precision final polish
    if time.time() < deadline - 5.0:
        def neg_value_hp(log_h):
            h = np.exp(log_h - np.mean(log_h))
            return -compute_value(h)
        try:
            res_hp = minimize(
                neg_value_hp,
                np.log(np.maximum(best_h, 1e-6)),
                method='L-BFGS-B',
                options={'maxiter': 10000, 'maxfun': 50000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            h_hp = np.exp(res_hp.x - np.mean(res_hp.x))
            h_hp = np.maximum(h_hp, 0.0)
            v_hp = compute_value(h_hp)
            if v_hp > best_val:
                best_val = v_hp
                best_h = h_hp.copy()
        except Exception:
            pass

    return list(np.asarray(best_h, dtype=float))
