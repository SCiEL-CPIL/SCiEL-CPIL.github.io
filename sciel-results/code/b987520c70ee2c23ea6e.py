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

def _get_sidon_set(n_elements, max_idx):
    """Generate a Sidon set of size n_elements within [0, max_idx).
    A Sidon set has all pairwise sums (with i<=j) distinct."""
    sidon = []
    used_sums = set()
    for i in range(max_idx):
        valid = True
        for j in range(len(sidon)):
            s = i + sidon[j]
            if s in used_sums:
                valid = False
                break
        if valid:
            s2 = 2 * i
            if s2 in used_sums:
                valid = False
        if valid:
            for j in range(len(sidon)):
                used_sums.add(i + sidon[j])
            used_sums.add(2 * i)
            sidon.append(i)
        if len(sidon) == n_elements:
            break
    return sidon

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.zeros(N)
    best_h[N // 2] = 1.0
    best_score = _score(best_h)
    
    from scipy.optimize import minimize
    
    # Generate Sidon sets of various sizes
    sidon_candidates = []
    for n_elem in [8, 10, 12, 14, 16, 18, 20]:
        s = _get_sidon_set(n_elem, N)
        if len(s) == n_elem:
            sidon_candidates.append(s)
    
    best_param = None
    best_param_score = 0.0
    
    for sidon in sidon_candidates:
        if time.time() > deadline - 30:
            break
        sidon_pos = list(sidon)
        n_active = len(sidon_pos)
        
        def neg_ratio_log(x, _sidon_pos=sidon_pos, _n_active=n_active):
            h = np.zeros(N)
            for i in range(_n_active):
                h[_sidon_pos[i]] = np.exp(x[i])
            if np.sum(h) < 1e-15:
                return 0.0
            return -_score(h)
        
        x0 = np.zeros(n_active)
        
        try:
            result = minimize(neg_ratio_log, x0, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            h_opt = np.zeros(N)
            for i in range(n_active):
                h_opt[sidon_pos[i]] = np.exp(result.x[i])
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
            if s_opt > best_param_score:
                best_param_score = s_opt
                best_param = (sidon_pos, result.x.copy())
        except Exception:
            pass
    
    # Polish best result
    if time.time() < deadline - 10 and best_param is not None:
        sidon_pos, x_best = best_param
        n_active = len(sidon_pos)
        
        def neg_ratio_polish(x, _sidon_pos=sidon_pos, _n_active=n_active):
            h = np.zeros(N)
            for i in range(_n_active):
                h[_sidon_pos[i]] = np.exp(x[i])
            if np.sum(h) < 1e-15:
                return 0.0
            return -_score(h)
        
        try:
            result = minimize(neg_ratio_polish, x_best, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            h_opt = np.zeros(N)
            for i in range(n_active):
                h_opt[sidon_pos[i]] = np.exp(result.x[i])
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final Nelder-Mead polish on full vector
    if time.time() < deadline - 5:
        def neg_score_full(h):
            h = np.abs(h)
            if np.sum(h) < 1e-15:
                return 0.0
            return -_score(h)
        
        try:
            result = minimize(neg_score_full, best_h, method='Nelder-Mead',
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
