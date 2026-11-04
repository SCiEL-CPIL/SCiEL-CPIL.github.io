"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _autoconv(h):
    """Compute autoconvolution of h on the grid [-1/4, 1/4]."""
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    return conv

def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or np.sum(h) < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    conv = _autoconv(h)
    sup = np.max(conv)
    if sup < 1e-30:
        return 0.0
    return l2_sq / (l1 * sup)

def _logR_and_grad(h, beta=100.0):
    """
    Compute log(ratio) and its gradient using a smooth approximation for the max.
    ratio = l2_sq / (l1 * sup)
    log(ratio) = log(l2_sq) - log(l1) - log(sup)
    We use log-sum-exp for a smooth max approximation.
    """
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 1e-15)  # avoid log(0)
    
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    conv = _autoconv(h)
    
    # Smooth max via log-sum-exp
    # log(sup) ≈ (1/beta) * log(sum(exp(beta * conv)))
    # But conv can have negative values, so we need to be careful
    # Actually, since h >= 0, conv >= 0 (autoconvolution of nonnegative is nonnegative)
    # Let's verify: conv[k] = sum_i h[i]*h[k-i] for valid indices, which is >= 0
    
    max_conv = np.max(conv)
    if max_conv < 1e-30:
        return -np.inf, np.zeros_like(h)
    
    # Shift for numerical stability
    conv_shifted = conv - max_conv
    log_sum_exp = np.log(np.sum(np.exp(beta * conv_shifted))) / beta + max_conv
    
    log_r = np.log(l2_sq) - np.log(l1) - log_sum_exp
    
    # Gradient of log_r with respect to h
    # d(log l2_sq)/dh_i = 2*h_i / l2_sq
    # d(log l1)/dh_i = 1 / l1
    # d(log_sum_exp)/dh_i: need chain rule through FFT
    # This is complex, so let's use numerical gradient for the max part
    # Or better: use the fact that d(log(sum(exp(beta*conv))))/dh = 
    #   beta * sum_k exp(beta*conv_k) * d(conv_k)/dh / sum_k exp(beta*conv_k)
    #   = beta * sum_k p_k * d(conv_k)/dh where p_k = exp(beta*conv_k)/sum_j exp(beta*conv_j)
    # And d(conv_k)/dh_i involves the convolution structure
    
    # For simplicity and speed, use numerical gradient for the full log_r
    eps = 1e-7
    grad = np.zeros_like(h)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        h_plus = np.maximum(h_plus, 1e-15)
        r_plus = _ratio(h_plus)
        
        h_minus = h.copy()
        h_minus[i] = max(0, h[i] - eps)
        r_minus = _ratio(h_minus)
        
        if r_plus > 0 and r_minus > 0:
            grad[i] = (np.log(r_plus) - np.log(r_minus)) / (2 * eps)
        elif r_plus > 0:
            grad[i] = np.log(r_plus) / eps
        else:
            grad[i] = 0.0
    
    return log_r, grad

def _local_optimize_smooth(h0, max_iters=500, tol=1e-10, beta=100.0):
    """Local optimization using projected gradient ascent on log(ratio)."""
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0)
    
    if np.sum(h) < 1e-15:
        h = np.ones(N) / N
    
    best_h = h.copy()
    best_r = _ratio(h)
    
    # Use scipy's minimize with gradient
    try:
        from scipy.optimize import minimize
        
        def neg_logR(h):
            h = np.maximum(h, 1e-15)
            val, _ = _logR_and_grad(h, beta)
            if val == -np.inf:
                return 1e10
            return -val
        
        def neg_logR_grad(h):
            h = np.maximum(h, 1e-15)
            val, grad = _logR_and_grad(h, beta)
            if val == -np.inf:
                return 1e10, np.zeros_like(h)
            return -val, -grad
        
        res = minimize(neg_logR, h, jac=neg_logR_grad, method='L-BFGS-B',
                      options={'maxiter': max_iters, 'gtol': tol, 'ftol': tol, 'maxls': 50})
        h_opt = np.maximum(res.x, 0)
        r_opt = _ratio(h_opt)
        if r_opt > best_r:
            best_h = h_opt
            best_r = r_opt
    except Exception:
        pass
    
    # Polish with Nelder-Mead on the actual ratio
    try:
        from scipy.optimize import minimize
        
        def neg_ratio(h):
            h = np.maximum(h, 0)
            return -_ratio(h)
        
        res = minimize(neg_ratio, best_h, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
        h_opt = np.maximum(res.x, 0)
        r_opt = _ratio(h_opt)
        if r_opt > best_r:
            best_h = h_opt
            best_r = r_opt
    except Exception:
        pass
    
    return best_h, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    grid_start = -0.25
    grid_end = 0.25
    
    best_h = np.ones(N)
    best_r = _ratio(best_h)
    
    # Phase 1: Structured parametric search
    centers = np.linspace(-0.2, 0.2, 9)
    widths = np.linspace(0.02, 0.25, 12)
    
    def _gaussian_profile(center, width, amplitude):
        x = np.linspace(grid_start, grid_end, N)
        return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
    
    def _lorentzian_profile(center, width, amplitude):
        x = np.linspace(grid_start, grid_end, N)
        return amplitude / (1 + ((x - center) / width) ** 2)
    
    for profile_func in [_gaussian_profile, _lorentzian_profile]:
        for c in centers:
            for w in widths:
                if time.time() > deadline:
                    break
                h = profile_func(c, w, 1.0)
                h = np.maximum(h, 0)
                if np.sum(h) < 1e-10:
                    continue
                r = _ratio(h)
                if r > best_r:
                    best_r = r
                    best_h = h.copy()
        
        if time.time() > deadline:
            break
    
    # Phase 2: Local optimization from best structured result
    if time.time() < deadline - 15:
        h_opt, r_opt = _local_optimize_smooth(best_h, max_iters=300)
        if r_opt > best_r:
            best_h = h_opt
            best_r = r_opt
    
    # Phase 3: Try additional local optimizations from diverse starting points
    if time.time() < deadline - 10:
        diverse_starts = []
        diverse_starts.append(np.ones(N))
        diverse_starts.append(np.abs(np.linspace(-1, 1, N)))
        h2 = np.zeros(N)
        h2[:N//4] = 1.0
        h2[3*N//4:] = 1.0
        diverse_starts.append(h2)
        # Narrow peak
        h3 = np.zeros(N)
        h3[N//2] = 1.0
        h3[N//2-1] = 0.5
        h3[N//2+1] = 0.5
        diverse_starts.append(h3)
        
        for h0 in diverse_starts:
            if time.time() > deadline - 5:
                break
            h_opt, r_opt = _local_optimize_smooth(h0, max_iters=150)
            if r_opt > best_r:
                best_h = h_opt
                best_r = r_opt
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 2:
        try:
            from scipy.optimize import minimize
            
            def neg_ratio(h):
                h = np.maximum(h, 0)
                return -_ratio(h)
            
            res = minimize(neg_ratio, best_h, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            h_opt = np.maximum(res.x, 0)
            r_opt = _ratio(h_opt)
            if r_opt > best_r:
                best_h = h_opt
                best_r = r_opt
        except Exception:
            pass
    
    # Ensure final result is valid
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return list(best_h)
# EVOLVE-BLOCK-END
