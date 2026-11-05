import numpy as np
import scipy.optimize as so
import time

N = 400

def _autoconv_max(h):
    """Compute the max of the full autoconvolution of h using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1 << (size - 1).bit_length()
    H = np.fft.rfft(h, n=fsize)
    conv = np.fft.irfft(H * H, n=fsize)[:size]
    return np.max(conv)

def _exact_value(h):
    """Compute the exact task value."""
    S = np.sum(h)
    if abs(S) < 1e-12:
        return 1e10
    M = _autoconv_max(h)
    return abs(2.0 * N * M) / (S * S)

def _build_h_from_params(params, seed=42):
    """
    Build h from parameters using a smoothed random-walk approach.
    params = [drift, step_scale, smooth_sigma]
    
    The idea: start with a random walk with negative drift (mean-reverting),
    then smooth it. This creates a smooth landscape for optimization.
    The drift pulls values back toward zero, keeping the signal bounded.
    """
    drift, step_scale, smooth_sigma = params
    rng = np.random.RandomState(seed)
    steps = rng.randn(N) * step_scale
    # Apply drift: each step is pulled back toward 0
    h = np.zeros(N)
    for i in range(N):
        h[i] = h[i-1] + drift * (0 - h[i-1]) + steps[i] if i > 0 else steps[i]
    
    # Smooth with a Gaussian kernel
    if smooth_sigma > 0.01:
        ksize = int(2 * smooth_sigma) | 1
        if ksize >= 3:
            x = np.arange(ksize) - ksize // 2
            kernel = np.exp(-x**2 / (2 * smooth_sigma**2))
            kernel /= kernel.sum()
            h = np.convolve(h, kernel, mode='same')
    
    # Normalize: shift so mean is 1 (to keep sum(h) ~ N)
    h = h - h.mean() + 1.0
    return h

def _objective(params, seed=42):
    """Objective: minimize the task value."""
    try:
        h = _build_h_from_params(params, seed)
        S = np.sum(h)
        if abs(S) < 1e-10:
            return 1e10
        M = _autoconv_max(h)
        val = abs(2.0 * N * M) / (S * S)
        return val
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Best found so far: the seed
    best_h = np.ones(N)
    best_val = _exact_value(best_h)  # Should be ~2.0
    
    # Parameter bounds: drift in [-1, 1], step_scale in [0, 5], smooth_sigma in [0, 20]
    bounds = [(-1.0, 1.0), (0.0, 5.0), (0.0, 20.0)]
    
    # Multiple initial guesses with different seeds for the random walk
    seeds = [42, 123, 7, 999, 555]
    inits = []
    for s in seeds:
        for drift in [-0.5, 0.0, 0.5]:
            for step_scale in [0.5, 1.0, 2.0]:
                for smooth in [0.0, 2.0, 5.0]:
                    inits.append([drift, step_scale, smooth])
    
    # Shuffle and limit number of optimizations
    np.random.shuffle(inits)
    max_opts = 30
    
    for idx, x0 in enumerate(inits[:max_opts]):
        if time.time() > deadline - 15:
            break
        
        seed = seeds[idx % len(seeds)]
        try:
            res = so.minimize(
                lambda p: _objective(p, seed),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_h = _build_h_from_params(res.x, seed)
        except Exception:
            pass
    
    # Final check
    final_val = _exact_value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h
