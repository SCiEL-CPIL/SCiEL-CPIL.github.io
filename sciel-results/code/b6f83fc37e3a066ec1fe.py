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
    
    # FFT based convolution for speed
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
    
    # l2 calculation: sum (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # where y = [0, g[0], ..., g[m-1], 0]
    # = (w/3) * (2*sum(g^2) + sum(g[:-1]*g[1:]))
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
    
    # Smooth max using log-sum-exp
    # max(g) ≈ (1/beta) * log(sum(exp(beta*g)))
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
    best_val = compute_value(best_h)  # should be 1.0
    
    rng = np.random.RandomState(42)
    
    def try_optimize_log(start_log_h, maxiter, use_smooth=True, beta=100.0):
        nonlocal best_h, best_val
        t0 = time.time()
        if use_smooth:
            obj = lambda x: neg_value_smooth(x, beta=beta)
        else:
            obj = neg_value_direct
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
        return time.time() - t0

    # Start from uniform (log 0 = 0)
    try_optimize_log(np.zeros(N), maxiter=5000, use_smooth=True)

    # Try multi-start perturbations if time remains
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        # Perturb: smooth random log-heights, centered so mean ~0
        x = rng.randn(N)
        # light smoothing
        kernel = np.ones(5)/5.0
        x = np.convolve(x, kernel, mode='same')
        x = x - x.mean()
        x = 0.5 * x + 0.5 * np.zeros(N)  # damp toward uniform
        # Add a few structured shapes
        if i == 1:
            x = np.sin(np.pi * np.arange(N) / N) + 0.3
        elif i == 2:
            x = np.cos(2*np.pi*np.arange(N)/N) * 0.5 + 0.5
        elif i == 3:
            x = 1 - 2*np.abs(np.arange(N)/N - 0.5)
        elif i == 4:
            x = np.exp(-((np.arange(N)-N//2)**2)/30.0)
            x = x - x.mean() + 0.5
        elif i == 5:
            x = np.where(np.arange(N) < N//2, 0.2, 0.8) + 0.1*rng.randn(N)
        elif i == 6:
            # Triangle peak
            x = 1 - 2*np.abs(np.arange(N)/N - 0.5)
            x = x**2
        elif i == 7:
            # Two peaks
            x = np.exp(-((np.arange(N)-N//4)**2)/10.0) + np.exp(-((np.arange(N)-3*N//4)**2)/10.0)
            x = x - x.mean() + 0.5
        elif i == 8:
            # Ramp
            x = np.linspace(0.2, 0.8, N)
        elif i == 9:
            # Alternating
            x = 0.5 + 0.3*np.cos(4*np.pi*np.arange(N)/N)
        elif i == 10:
            # Narrow central peak
            x = np.exp(-((np.arange(N)-N//2)**2)/5.0)
            x = x - x.mean() + 0.5
        elif i == 11:
            # Broad central peak
            x = np.exp(-((np.arange(N)-N//2)**2)/80.0)
            x = x - x.mean() + 0.5
        elif i == 12:
            # U-shape
            x = 2*np.abs(np.arange(N)/N - 0.5)
            x = x - x.mean() + 0.5
        elif i == 13:
            # Sine with higher frequency
            x = 0.5 + 0.4*np.sin(3*np.pi*np.arange(N)/N)
        elif i == 14:
            # Plateau
            x = np.where((np.arange(N) > N//4) & (np.arange(N) < 3*N//4), 0.9, 0.3)
        elif i == 15:
            # Raised cosine (Hann) window — reduces autocorrelation peak
            x = 0.5 * (1.0 + np.cos(2.0 * np.pi * np.arange(N) / N))
        elif i == 16:
            # Double sine
            x = 0.5 + 0.3*np.sin(2*np.pi*np.arange(N)/N) + 0.2*np.sin(4*np.pi*np.arange(N)/N)
        elif i == 17:
            # Gaussian mixture with offset
            x = 0.6*np.exp(-((np.arange(N)-N//3)**2)/15.0) + 0.4*np.exp(-((np.arange(N)-2*N//3)**2)/15.0)
            x = x - x.mean() + 0.5
        elif i == 18:
            # Flat-top with tapered edges
            x = np.ones(N)
            taper = 8
            for k in range(taper):
                frac = 0.5*(1.0 + np.cos(np.pi*(k+0.5)/taper))
                x[k] = frac
                x[N-1-k] = frac
        else:
            x = x + 0.2*rng.randn(N)
        x = np.maximum(x, -5.0)  # avoid underflow
        try_optimize_log(x, maxiter=3000, use_smooth=True)

    # Final polish: direct optimization from best found
    if time.time() < deadline - 25.0:
        try_optimize_log(np.log(np.maximum(best_h, 1e-6)), maxiter=8000, use_smooth=False)

    # One more log-space polish
    if time.time() < deadline - 10.0:
        try_optimize_log(np.log(np.maximum(best_h, 1e-6)), maxiter=5000, use_smooth=True, beta=200.0)

    # High-precision final polish: tighter tolerances, more iterations
    if time.time() < deadline - 5.0:
        def neg_value_hp(log_h):
            h = np.exp(log_h - np.mean(log_h))
            return -compute_value(h)
        t0 = time.time()
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

    return list(np.asarray(best_h, dtype=float))
