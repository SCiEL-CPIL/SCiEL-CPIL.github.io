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
    
    # Compute autoconvolution peak efficiently
    # f is a step function: f(x) = h[i] for x in [center_i - w/2, center_i + w/2)
    # centers: -1/4 + (i+0.5)*w
    # (f*f)(t) = integral f(s) f(t-s) ds
    # For step functions, this can be computed via the overlap of intervals
    
    # Use FFT approach: sample f on a fine grid, convolve
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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_h = np.zeros(N)
    best_h[N // 2] = 1.0
    best_score = _score(best_h)
    
    # Generate diverse candidates
    candidates = []
    
    # Single spikes at various positions
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        candidates.append(h)
    
    # Two spikes
    for i in range(0, N, 5):
        for j in range(i+1, N, 5):
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            candidates.append(h)
    
    # Triangular profiles
    for width in [2, 4, 6, 10, 20]:
        half = width // 2
        for center in range(half, N - half):
            h = np.zeros(N)
            for i in range(N):
                dist = abs(i - center)
                if dist <= half:
                    h[i] = 1.0 - dist / half
            if np.sum(h) > 0:
                candidates.append(h)
    
    # Gaussian profiles
    for sigma in [1, 2, 3, 5, 8]:
        for center in [N//4, N//2, 3*N//4]:
            i = np.arange(N)
            h = np.exp(-0.5 * ((i - center) / sigma)**2)
            candidates.append(h)
    
    # Uniform blocks
    for width in [1, 2, 5, 10, 25, 50]:
        for start in range(0, N - width + 1, max(1, (N - width) // 10)):
            h = np.zeros(N)
            h[start:start+width] = 1.0
            candidates.append(h)
    
    # Evaluate candidates
    for h in candidates:
        if time.time() > deadline - 10:
            break
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization from best candidate
    if time.time() < deadline - 20:
        from scipy.optimize import minimize
        
        def neg_score(h):
            h = np.abs(h)
            return -_score(h)
        
        try:
            result = minimize(neg_score, best_h, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10,
                                    'adaptive': True})
            h_opt = np.abs(result.x)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
        except Exception:
            pass
        
        # Second round of optimization from a different starting point
        if time.time() < deadline - 10:
            h_start = np.zeros(N)
            h_start[N//2] = 1.0
            h_start[N//2 - 1] = 0.5
            h_start[N//2 + 1] = 0.5
            try:
                result = minimize(neg_score, h_start, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                h_opt = np.abs(result.x)
                s_opt = _score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt
            except Exception:
                pass
    
    return list(best_h)
