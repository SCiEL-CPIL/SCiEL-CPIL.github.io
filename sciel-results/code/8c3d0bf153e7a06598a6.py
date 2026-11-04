"""Second autocorrelation inequality: maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf).

Strategy: Use a DFT spectral parameterization h = |IDFT(z)|^2 with sparse
non-negative spectral support to naturally enforce non-negativity and exploit
the spectral structure of the autoconvolution. Combine with gradient-based
optimization (L-BFGS-B) on the complex parameters and a smooth-max approximation
for the supremum norm.
"""
import numpy as np
import time

N = 50
GRID_START = -0.25
GRID_END = 0.25
DX = (GRID_END - GRID_START) / (N - 1)

def _autoconv(h):
    """Compute autoconvolution of h on the grid using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    return conv

def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12) or np.sum(h) < 1e-15:
        return 0.0
    h = np.maximum(h, 0)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    conv = _autoconv(h)
    sup = np.max(conv)
    if sup < 1e-30:
        return 0.0
    return l2_sq / (l1 * sup)

def _smooth_max(x, alpha=50.0):
    """Smooth approximation of max(x) using log-sum-exp."""
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    if m < 1e-30:
        return 1e-30
    return m + np.log(np.sum(np.exp(alpha * (x - m)))) / alpha

def _h_from_spectral(z):
    """
    Given complex vector z of length N, compute h = |IDFT(z)|^2.
    This ensures h >= 0 automatically.
    """
    # IDFT of z gives a complex sequence; take magnitude squared
    x = np.fft.ifft(z)
    h = np.abs(x) ** 2
    return h

def _objective_complex(params):
    """
    Objective in terms of complex spectral parameters.
    params: real vector of length 2*K (magnitudes and phases for K bins)
    Returns negative log of the ratio for minimization.
    """
    K = params.size // 2
    mags = params[:K]
    phases = params[K:]
    
    # Build sparse spectral vector
    z = np.zeros(N, dtype=complex)
    for k in range(K):
        z[k] = mags[k] * np.exp(1j * phases[k])
    
    h = _h_from_spectral(z)
    h = np.maximum(h, 0)
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    
    l2_sq = np.sum(h * h)
    conv = _autoconv(h)
    sup = _smooth_max(conv, alpha=30.0)
    
    if sup < 1e-30:
        return 1e10
    
    ratio = l2_sq / (l1 * sup)
    if ratio <= 0:
        return 1e10
    
    return -np.log(ratio)

def _objective_direct(h):
    """Direct objective: negative log of ratio."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    l2_sq = np.sum(h * h)
    conv = _autoconv(h)
    sup = _smooth_max(conv, alpha=30.0)
    if sup < 1e-30:
        return 1e10
    ratio = l2_sq / (l1 * sup)
    if ratio <= 0:
        return 1e10
    return -np.log(ratio)

def _objective_direct_jac(h):
    """Objective and gradient for direct h-space optimization."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10, np.zeros(N)
    
    l2_sq = np.sum(h * h)
    conv = _autoconv(h)
    alpha = 30.0
    m = np.max(conv)
    if m < 1e-30:
        return 1e10, np.zeros(N)
    
    # Smooth max and its gradient
    exp_terms = np.exp(alpha * (conv - m))
    sum_exp = np.sum(exp_terms)
    sup = m + np.log(sum_exp) / alpha
    # d(sup)/d(conv_i) = exp(alpha*(conv_i - m)) / sum_exp
    d_sup_d_conv = exp_terms / sum_exp
    
    ratio = l2_sq / (l1 * sup)
    if ratio <= 0:
        return 1e10, np.zeros(N)
    
    obj = -np.log(ratio)
    
    # Gradient: d(-log(ratio))/d(h_i)
    # ratio = l2_sq / (l1 * sup)
    # d(log(ratio))/d(h_i) = d(log(l2_sq))/d(h_i) - d(log(l1))/d(h_i) - d(log(sup))/d(h_i)
    # = 2*h_i/l2_sq - 1/l1 - (1/sup) * d(sup)/d(h_i)
    
    # d(sup)/d(h_i) = sum_j d(sup)/d(conv_j) * d(conv_j)/d(h_i)
    # conv_j = sum_k h_k * h_{j-k} (discrete)
    # This is complex; use finite differences for gradient
    
    # For simplicity, use finite difference gradient
    eps = 1e-7
    grad = np.zeros(N)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] = max(0, h[i] + eps)
        obj_plus = _objective_direct(h_plus)
        h_minus = h.copy()
        h_minus[i] = max(0, h[i] - eps)
        obj_minus = _objective_direct(h_minus)
        grad[i] = (obj_plus - obj_minus) / (2 * eps)
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    best_h = np.ones(N)
    best_r = _ratio(best_h)
    
    # Phase 1: DFT spectral parameterization optimization
    # Try different numbers of active bins
    K_values = [1, 2, 3, 4, 5]
    
    from scipy.optimize import minimize
    
    for K in K_values:
        if time_left() < 15:
            break
        
        # Initial guess: spread the K bins
        initial_mags = np.ones(K) * 0.5
        initial_phases = np.linspace(0, 2*np.pi, K, endpoint=False)
        params0 = np.concatenate([initial_mags, initial_phases])
        
        def neg_log_ratio(params):
            return _objective_complex(params)
        
        try:
            res = minimize(neg_log_ratio, params0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < _objective_complex(params0):
                h_opt = _h_from_spectral(np.zeros(N, dtype=complex))
                # Reconstruct from optimized params
                mags_opt = np.maximum(res.x[:K], 0)
                phases_opt = res.x[K:]
                z = np.zeros(N, dtype=complex)
                for k in range(K):
                    z[k] = mags_opt[k] * np.exp(1j * phases_opt[k])
                h_opt = _h_from_spectral(z)
                h_opt = np.maximum(h_opt, 0)
                r_opt = _ratio(h_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Direct optimization in h-space from best result
    if time_left() > 20:
        try:
            def neg_ratio_direct(h):
                return _objective_direct(h)
            
            res = minimize(neg_ratio_direct, best_h, method='L-BFGS-B',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            h_opt = np.maximum(res.x, 0)
            r_opt = _ratio(h_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Try diverse structured initializations
    if time_left() > 10:
        diverse_starts = []
        
        # Uniform
        diverse_starts.append(np.ones(N))
        
        # Triangular
        diverse_starts.append(np.abs(np.linspace(-1, 1, N)))
        
        # Two peaks
        h2 = np.zeros(N)
        h2[:N//4] = 1.0
        h2[3*N//4:] = 1.0
        diverse_starts.append(h2)
        
        # Gaussian
        x = np.linspace(GRID_START, GRID_END, N)
        diverse_starts.append(np.exp(-50 * x**2))
        
        # Lorentzian
        diverse_starts.append(1.0 / (1 + 20 * x**2))
        
        # Cosine bump
        diverse_starts.append(0.5 * (1 + np.cos(2*np.pi*x)))
        
        for h0 in diverse_starts:
            if time_left() < 8:
                break
            try:
                def neg_ratio_h0(h):
                    return _objective_direct(h)
                
                res = minimize(neg_ratio_h0, h0, method='L-BFGS-B',
                              bounds=[(0, None)] * N,
                              options={'maxiter': 100, 'ftol': 1e-10})
                h_opt = np.maximum(res.x, 0)
                r_opt = _ratio(h_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 4: Coordinate descent polish
    if time_left() > 5:
        h = best_h.copy()
        for _ in range(20):
            if time_left() < 3:
                break
            improved = False
            for i in range(N):
                if time_left() < 3:
                    break
                h_save = h[i]
                # Try a few perturbations
                for delta in [-0.05, -0.02, 0.02, 0.05]:
                    h[i] = max(0, h_save + delta)
                    r = _ratio(h)
                    if r > best_r:
                        best_r = r
                        best_h = h.copy()
                        improved = True
                h[i] = h_save
            if not improved:
                break
    
    # Ensure valid output
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return list(best_h)
