import numpy as np
import time
import scipy.optimize as opt

N = 600
TIME_LIMIT = 140

def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p

def _fft_autoconv(f):
    """Compute full autoconvolution via FFT."""
    n = len(f)
    size = 2 * n - 1
    fft_size = _next_pow2(size)
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    return conv[:size]

def _compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = _fft_autoconv(f)
    return np.max(conv) / (s * s)

def _soft_max(conv, beta=50.0):
    """Smooth approximation of max."""
    conv = np.asarray(conv)
    if conv.size == 0:
        return 0.0
    c_max = np.max(conv)
    if c_max < 1e-15:
        return 0.0
    # Shift to avoid overflow
    shifted = conv - c_max
    return c_max + np.log(np.sum(np.exp(beta * shifted))) / beta

def _parametric_objective(params):
    """
    Objective function for parametric optimization.
    params: [amp1, mu1, sigma1, amp2, mu2, sigma2, ...]
    We use a sum of Gaussians to allow flexibility.
    """
    # Number of Gaussians
    n_gauss = 3
    # Unpack parameters
    amps = params[0:n_gauss]
    mus = params[n_gauss:2*n_gauss]
    sigmas = params[2*n_gauss:3*n_gauss]
    
    # Ensure sigmas are positive
    sigmas = np.abs(sigmas) + 1e-6
    # Ensure amps are positive
    amps = np.abs(amps) + 1e-6
    
    # Grid
    # The interval is [-1/4, 1/4]
    # N intervals, so dx = (0.5) / N
    # Centers of intervals
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
    
    # Construct f
    f = np.zeros(N)
    for i in range(n_gauss):
        f += amps[i] * np.exp(-0.5 * ((x - mus[i]) / sigmas[i])**2)
        
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
        
    # Normalize f to sum to 1? 
    # The score is max(conv) / (sum(f))^2.
    # Since we are minimizing this ratio, the scale of f doesn't matter.
    # We can fix the scale to simplify optimization, e.g., sum(f)=1.
    # But the parametric form allows arbitrary scale.
    # Let's just compute the score directly.
    
    conv = _fft_autoconv(f)
    max_conv = _soft_max(conv, beta=50.0)
    
    # Add a penalty for non-negativity violation? 
    # Our construction is inherently non-negative (sum of positive Gaussians).
    # So no penalty needed.
    
    return max_conv / (s * s)

def _local_refine_discrete(f_init, start_time, max_time=120.0):
    """
    Fast discrete coordinate descent.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = _compute_score(f)
    
    # Precompute FFT size
    fft_size = _next_pow2(2 * N - 1)
    
    for sweep in range(20):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        # Shuffle indices to avoid bias
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() - start_time > max_time:
                break
            
            old_val = f[i]
            S = np.sum(f)
            
            # Compute background conv (with f[i]=0)
            # This is expensive if done from scratch every time.
            # But N=600 is small.
            f_bg = f.copy()
            f_bg[i] = 0.0
            conv_bg = _fft_autoconv(f_bg)
            
            # Affected lags
            m_start = max(0, i)
            m_end = min(2 * N - 2, i + N - 1)
            js = np.arange(m_start, m_end + 1) - i
            
            conv_bg_affected = conv_bg[m_start:m_end + 1]
            f_js = f[js]
            mask_diag = (js == i)
            f_js_offdiag = f_js[~mask_diag]
            
            def objective(v):
                if v < 0:
                    return np.inf
                S_new = S - old_val + v
                if S_new < 1e-15:
                    return np.inf
                
                new_conv = conv_bg_affected.copy()
                new_conv[mask_diag] += v * v
                new_conv[~mask_diag] += 2 * v * f_js_offdiag
                
                max_conv = np.max(new_conv)
                return max_conv / (S_new * S_new)
            
            # Try a few candidate values
            # Use a small grid around old_val
            # Also try 0 and some large values
            candidates_v = [0.0, old_val * 0.5, old_val, old_val * 1.5, old_val * 2.0, 1.0 / N, 2.0 / N, 5.0 / N]
            # Add random perturbations
            for _ in range(3):
                candidates_v.append(old_val * (1.0 + 0.2 * np.random.randn()))
            
            best_v = old_val
            best_obj = objective(old_val)
            for v in candidates_v:
                if v < 0 or v > 10.0: # Cap value
                    continue
                val = objective(v)
                if val < best_obj:
                    best_obj = val
                    best_v = v
            
            if best_v != old_val:
                f[i] = best_v
                improved = True
        
        s = np.sum(f)
        if s > 1e-15:
            f = f / s
        
        current_val = _compute_score(f)
        if current_val < best_val - 1e-12:
            best_val = current_val
            best_f = f.copy()
        
        if not improved:
            break
    
    return best_f

def construct():
    start_time = time.time()
    
    # 1. Parametric Optimization
    # Start with a single Gaussian centered at 0
    # params: [amp, mu, sigma]
    # amp=1, mu=0, sigma=0.1
    x0 = np.array([1.0, 0.0, 0.1])
    
    best_f = None
    best_score = np.inf
    
    # Try a few initializations
    inits = [
        np.array([1.0, 0.0, 0.1]),
        np.array([1.0, 0.0, 0.2]),
        np.array([1.0, 0.0, 0.05]),
        np.array([1.0, 0.1, 0.1]),
        np.array([1.0, -0.1, 0.1]),
    ]
    
    for x0 in inits:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        try:
            res = opt.minimize(_parametric_objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 100, 'maxfun': 1000})
            
            if res.success or res.fun < best_score:
                # Convert params back to discrete f
                amps = np.abs(res.x[0:1]) + 1e-6
                mus = res.x[1:2]
                sigmas = np.abs(res.x[2:3]) + 1e-6
                
                x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
                f = np.zeros(N)
                f += amps[0] * np.exp(-0.5 * ((x - mus[0]) / sigmas[0])**2)
                f = np.maximum(f, 0.0)
                
                score = _compute_score(f)
                if score < best_score:
                    best_score = score
                    best_f = f.copy()
                    
        except Exception as e:
            pass

    # If no parametric solution found, use a default
    if best_f is None:
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
        best_f = np.exp(-10 * x**2)
        best_f = best_f / np.sum(best_f)
        best_score = _compute_score(best_f)

    # 2. Discrete Refinement
    if time.time() - start_time < TIME_LIMIT - 10:
        remaining = TIME_LIMIT - 5 - (time.time() - start_time)
        if remaining > 5:
            refined_f = _local_refine_discrete(best_f, start_time, remaining)
            refined_score = _compute_score(refined_f)
            if refined_score < best_score:
                best_f = refined_f
                best_score = refined_score

    # Final normalization
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]
