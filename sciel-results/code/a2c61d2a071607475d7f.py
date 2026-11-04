import numpy as np
import time

N = 50
W = 1.0 / (2.0 * N)  # interval width = 0.01

def _score(h):
    """Compute ||f||_2^2 / (||f||_1 * ||f*f||_inf) for step heights h."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    w = W
    
    l2_sq = np.sum(h**2) * w
    l1 = np.sum(h) * w
    
    if l1 < 1e-15:
        return 0.0
    
    # Compute autoconvolution peak efficiently via FFT
    n_grid = 4 * n * 8  # oversample for accuracy
    t = np.linspace(-0.5, 0.5, n_grid, endpoint=False)
    dt = t[1] - t[0]
    
    # Build f on grid
    f_grid = np.zeros(n_grid)
    centers = -0.25 + (np.arange(n) + 0.5) * w
    for i in range(n):
        lo = centers[i] - w/2
        hi = centers[i] + w/2
        idx_lo = int((lo + 0.5) / dt)
        idx_hi = int((hi + 0.5) / dt)
        idx_lo = max(0, min(idx_lo, n_grid))
        idx_hi = max(0, min(idx_hi, n_grid))
        if idx_hi > idx_lo:
            f_grid[idx_lo:idx_hi] += h[i]
    
    # Convolve via FFT
    f_fft = np.fft.fft(f_grid, n=2*n_grid)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))[:n_grid] * dt
    
    max_conv = np.max(conv)
    
    if max_conv < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * max_conv)

def _two_lobe_heights(sep, sigma1, sigma2, a, b):
    """Generate heights as |a*w1 + b*w2|^2 where w1, w2 are disjoint Gaussian pulses."""
    k = np.arange(N)
    center = N / 2.0
    # Two lobes separated by 'sep' indices
    c1 = center - sep / 2.0
    c2 = center + sep / 2.0
    w1 = np.exp(-0.5 * ((k - c1) / sigma1)**2)
    w2 = np.exp(-0.5 * ((k - c2) / sigma2)**2)
    # Sum of amplitudes (real, so |a*w1 + b*w2|^2 = (a*w1 + b*w2)^2)
    h = (a * w1 + b * w2)**2
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_h = np.zeros(N)
    best_h[N // 2] = 1.0
    best_score = _score(best_h)
    
    # Strategy: two-lobe parametric search
    # h_k = (a * w1(k) + b * w2(k))^2 where w1, w2 are Gaussian pulses
    # Optimize: sep (separation), sigma1, sigma2, a, b
    from scipy.optimize import minimize
    
    def param_to_h(params):
        sep, sigma1, sigma2, a, b = params
        sep = np.clip(sep, 1.0, N - 2)
        sigma1 = np.clip(sigma1, 0.5, N/2)
        sigma2 = np.clip(sigma2, 0.5, N/2)
        a = abs(a)
        b = abs(b)
        return _two_lobe_heights(sep, sigma1, sigma2, a, b)
    
    def neg_param_score(params):
        h = param_to_h(params)
        if np.sum(h) < 1e-15:
            return 0.0
        return -_score(h)
    
    # Global search over parameter space
    best_param = None
    best_param_score = 0.0
    
    # Grid search over separation and widths
    for sep in [4, 6, 8, 10, 12, 14, 16, 20, 24, 30, 36, 40]:
        for sigma1 in [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
            for sigma2 in [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
                for a in [0.5, 1.0, 1.5, 2.0]:
                    for b in [0.5, 1.0, 1.5, 2.0]:
                        if time.time() > deadline - 30:
                            break
                        params = [sep, sigma1, sigma2, a, b]
                        h = param_to_h(params)
                        s = _score(h)
                        if s > best_score:
                            best_score = s
                            best_h = h.copy()
                        if s > best_param_score:
                            best_param_score = s
                            best_param = params.copy()
    
    # Local optimization from best parameter found
    if time.time() < deadline - 15 and best_param is not None:
        try:
            result = minimize(neg_param_score, best_param, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                    'adaptive': True})
            h_opt = param_to_h(result.x)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
        except Exception:
            pass
    
    # Try asymmetric amplitudes
    if time.time() < deadline - 10:
        for sep in [8, 12, 16, 20, 24]:
            for sigma in [2.0, 3.0, 4.0, 6.0]:
                for ratio in [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]:
                    a = 1.0
                    b = ratio
                    params = [sep, sigma, sigma, a, b]
                    h = param_to_h(params)
                    s = _score(h)
                    if s > best_score:
                        best_score = s
                        best_h = h.copy()
    
    # Final polish: direct optimization on h from best result
    if time.time() < deadline - 5:
        def neg_score(h):
            h = np.abs(h)
            if np.sum(h) < 1e-15:
                return 0.0
            return -_score(h)
        
        try:
            result = minimize(neg_score, best_h, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11,
                                    'adaptive': True})
            h_opt = np.abs(result.x)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
        except Exception:
            pass
    
    return list(best_h)
