"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 600

# EVOLVE-BLOCK-START

def _midpoints():
    """Midpoints of the 600 equal intervals on [-1/4, 1/4]."""
    dx = (0.25 - (-0.25)) / N
    return -0.25 + dx * (np.arange(N) + 0.5)

def _autoconv_fft(heights):
    """Compute linear autoconvolution via FFT. Returns array of length 2N-1."""
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, n=size)
    acorr = np.fft.irfft(H * np.conj(H), n=size)
    # acorr has length `size`, but linear autoconvolution has length 2n-1
    # The first n values are the positive lags, but we need the full convolution
    # Actually for autoconvolution (not autocorrelation), we want f*f
    # Let me use full convolution instead
    f = np.zeros(size)
    f[:n] = h
    F = np.fft.rfft(f, n=size)
    conv = np.fft.irfft(F * F, n=size)
    return conv[:2*n-1]

def _compute_score(heights):
    """
    Compute the normalized max of autoconvolution.
    
    The metric: max(f*f) / (∫f)^2 * W where W is the support width,
    normalized so that the uniform distribution on [-1/4,1/4] gives 1.0.
    
    For uniform f=1 on [-1/4,1/4] with width W=1/2:
    ∫f = W = 1/2, (∫f)^2 = 1/4
    (f*f)(0) = W = 1/2
    max(f*f)/(∫f)^2 = 2
    
    So to get 1.0 for uniform, we need: max(f*f) * W / (∫f)^2
    = (1/2) * (1/2) / (1/4) = 1.0 ✓
    
    In discrete terms with dx = W/N:
    ∫f ≈ dx * sum(h)
    (f*f)(k) ≈ dx * sum_i h[i]*h[i+k] (for the discrete convolution)
    max(f*f) ≈ dx * max(conv) where conv is the discrete autoconvolution
    
    Score = (dx * max(conv)) * W / (dx * sum(h))^2
           = W * max(conv) / (dx * (sum(h))^2)
           = W * max(conv) * N / (W * (sum(h))^2)
           = max(conv) * N / (sum(h))^2
    """
    h = np.asarray(heights, dtype=np.float64)
    if np.all(h == 0):
        return float('inf')
    
    # Compute autoconvolution
    conv = _autoconv_fft(h)
    
    total = np.sum(h)
    if total == 0:
        return float('inf')
    
    max_conv = np.max(conv)
    score = max_conv * N / (total * total)
    return score

def _gaussian_mixture_to_heights(params, x):
    """
    Convert Gaussian mixture parameters to step heights.
    params: array of shape (K, 3) where each row is (amplitude, center, width)
    x: array of midpoints
    Returns: nonnegative heights
    """
    K = len(params)
    heights = np.zeros(len(x))
    for k in range(K):
        amp, center, width = params[k]
        if width <= 1e-10:
            width = 1e-10
        heights += amp * np.exp(-0.5 * ((x - center) / width) ** 2)
    return np.maximum(heights, 0.0)

def _smooth_max_surrogate(conv, temperature=0.01):
    """Smooth max using log-sum-exp approximation."""
    max_val = np.max(conv)
    if temperature <= 0:
        return max_val
    shifted = conv / temperature
    shifted = shifted - np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + temperature * log_sum

def _objective(params, x, use_smooth=True, temperature=0.01):
    """Objective function to minimize."""
    heights = _gaussian_mixture_to_heights(params, x)
    total = np.sum(heights)
    if total < 1e-10:
        return 1e10
    
    conv = _autoconv_fft(heights)
    
    if use_smooth:
        max_conv = _smooth_max_surrogate(conv, temperature)
    else:
        max_conv = np.max(conv)
    
    score = max_conv * N / (total * total)
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    x = _midpoints()
    
    # Best result so far
    best_heights = None
    best_score = float('inf')
    
    def _update_best(heights, score):
        nonlocal best_heights, best_score
        if score < best_score:
            best_heights = heights.copy()
            best_score = score
    
    # Initial candidates: various parametric forms
    
    # 1. Uniform
    h_uniform = np.ones(N)
    s_uniform = _compute_score(h_uniform)
    _update_best(h_uniform, s_uniform)
    
    # 2. Single Gaussian centered
    for sigma in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        h_gauss = np.exp(-0.5 * (x / sigma) ** 2)
        s_gauss = _compute_score(h_gauss)
        _update_best(h_gauss, s_gauss)
    
    # 3. Raised cosine
    for a in [0.5, 0.7, 1.0, 1.5, 2.0]:
        h_cos = 0.5 * (1 + np.cos(np.pi * x / a))
        h_cos = np.maximum(h_cos, 0)
        if np.sum(h_cos) > 0:
            s_cos = _compute_score(h_cos)
            _update_best(h_cos, s_cos)
    
    # 4. Flat-top with tapered edges
    for taper_frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
        h_flat = np.ones(N)
        n_taper = int(N * taper_frac / 2)
        if n_taper > 0:
            # Taper left edge
            h_flat[:n_taper] = np.linspace(0, 1, n_taper)
            # Taper right edge
            h_flat[-n_taper:] = np.linspace(1, 0, n_taper)
        s_flat = _compute_score(h_flat)
        _update_best(h_flat, s_flat)
    
    # 5. Try optimizing a 2-component Gaussian mixture
    if time.time() < deadline - 60:
        # Initial guess: two Gaussians at -0.1 and +0.1
        x0 = np.array([
            [1.0, -0.1, 0.1],   # amplitude, center, width
            [1.0, 0.1, 0.1]
        ])
        
        def obj(params):
            if time.time() > deadline - 5:
                return 1e10
            return _objective(params, x, use_smooth=True, temperature=0.01)
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B', 
                             bounds=[(0.01, 10), (-0.25, 0.25), (0.01, 0.3)] * 2,
                             options={'maxiter': 200, 'ftol': 1e-8})
            if result.success:
                h_opt = _gaussian_mixture_to_heights(result.x, x)
                s_opt = _compute_score(h_opt)
                _update_best(h_opt, s_opt)
        except Exception:
            pass
    
    # 6. Try 3-component Gaussian mixture
    if time.time() < deadline - 40:
        x0_3 = np.array([
            [0.8, -0.15, 0.08],
            [1.2, 0.0, 0.1],
            [0.8, 0.15, 0.08]
        ])
        
        def obj3(params):
            if time.time() > deadline - 5:
                return 1e10
            return _objective(params, x, use_smooth=True, temperature=0.005)
        
        try:
            result3 = minimize(obj3, x0_3, method='L-BFGS-B',
                              bounds=[(0.01, 10), (-0.25, 0.25), (0.01, 0.3)] * 3,
                              options={'maxiter': 300, 'ftol': 1e-8})
            if result3.success:
                h_opt3 = _gaussian_mixture_to_heights(result3.x, x)
                s_opt3 = _compute_score(h_opt3)
                _update_best(h_opt3, s_opt3)
        except Exception:
            pass
    
    # 7. Local refinement of best height vector using coordinate descent
    if best_heights is not None and time.time() < deadline - 20:
        h_refine = best_heights.copy()
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            improved = False
            for idx in range(N):
                old_val = h_refine[idx]
                # Try a few perturbations
                for delta in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                    h_refine[idx] = max(0.0, old_val + delta)
                    s_test = _compute_score(h_refine)
                    if s_test < best_score - 1e-10:
                        best_score = s_test
                        improved = True
                        break
                if not improved:
                    h_refine[idx] = old_val
            if not improved:
                break
        _update_best(h_refine, best_score)
    
    if best_heights is None:
        return list(np.ones(N))
    
    return list(best_heights)

# EVOLVE-BLOCK-END
