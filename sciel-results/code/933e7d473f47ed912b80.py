import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for circular autoconvolution."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    # Circular autoconvolution
    hh = np.convolve(h, h, mode='full')[:N]  # circular
    # Actually, for circular: use FFT
    hh = np.fft.ifft(np.fft.fft(h) * np.fft.fft(h)).real
    sup = np.max(hh)
    if sup < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup)

def _smooth_objective(params, smoothness=10.0):
    """Negative of smooth approximation of the ratio for minimization."""
    h = np.exp(params)  # ensures nonnegativity
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    hh = np.fft.ifft(np.fft.fft(h) * np.fft.fft(h)).real
    # Smooth max via log-sum-exp
    m = np.max(hh)
    if m < 1e-15:
        return 0.0
    logsumexp = m + np.log(np.sum(np.exp((hh - m) / smoothness))) * smoothness
    # Actually: log(sum(exp(x/s))) * s approximates max(x) when s is small
    # Let me use: soft_max = s * log(sum(exp(x/s)))
    s = smoothness
    shifted = hh / s
    shifted -= np.max(shifted)
    soft_max = s * np.log(np.sum(np.exp(shifted)))
    if soft_max < 1e-15:
        return 0.0
    ratio = l2_sq / (l1 * soft_max)
    return -ratio

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin
    
    best_ratio = 0.0
    best_h = np.ones(N)
    
    # Initial: all ones
    best_ratio = _compute_ratio(best_h)
    
    # Multi-start L-BFGS-B
    rng = np.random.default_rng(42)
    
    def make_initialization(idx):
        if idx == 0:
            return np.ones(N)
        elif idx == 1:
            # Gaussian centered
            x = np.linspace(-1, 1, N)
            return np.exp(-5 * x**2)
        elif idx == 2:
            # Gaussian off-center
            x = np.linspace(-1, 1, N)
            return np.exp(-5 * (x - 0.5)**2)
        elif idx == 3:
            # Bimodal
            x = np.linspace(-1, 1, N)
            return np.exp(-10 * (x - 0.3)**2) + np.exp(-10 * (x + 0.3)**2)
        elif idx == 4:
            # Single spike
            h = np.zeros(N)
            h[N//2] = 1.0
            return h
        elif idx == 5:
            # Random smooth
            h = rng.random(N)
            # Smooth it
            from scipy.ndimage import gaussian_filter1d
            h = gaussian_filter1d(h, sigma=2.0)
            return h
        else:
            # Random
            return rng.random(N) + 0.1
    
    n_starts = 12
    
    for i in range(n_starts):
        if time.time() > deadline - 5.0:
            break
        
        h0 = make_initialization(i)
        h0 = np.maximum(h0, 1e-6)  # ensure positive
        
        # Parameterize as log to ensure nonnegativity
        params0 = np.log(h0)
        
        try:
            result = minimize(
                _smooth_objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_opt = np.exp(result.x)
            ratio = _compute_ratio(h_opt)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_opt.copy()
                
        except Exception:
            continue
    
    # Try different smoothness values for best result
    if time.time() < deadline - 5.0:
        for s in [1.0, 0.5, 0.1, 0.05]:
            if time.time() > deadline - 3.0:
                break
            params0 = np.log(np.maximum(best_h, 1e-6))
            try:
                result = minimize(
                    _smooth_objective,
                    params0,
                    args=(s,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                h_opt = np.exp(result.x)
                ratio = _compute_ratio(h_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_h = h_opt.copy()
            except Exception:
                continue
    
    # Final polish with very small smoothness
    if time.time() < deadline - 2.0:
        params0 = np.log(np.maximum(best_h, 1e-6))
        try:
            result = minimize(
                _smooth_objective,
                params0,
                args=(0.01,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            h_opt = np.exp(result.x)
            ratio = _compute_ratio(h_opt)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
