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

def _softplus_project(f, temperature=0.01):
    """Project to non-negative via softplus-like mapping, preserving sum."""
    f = np.asarray(f, dtype=np.float64)
    s = np.sum(f)
    if s < 1e-15:
        return np.ones(N) / N
    # Normalize to mean 1, then apply softplus
    f_norm = f / s * N
    # Softplus: log(1 + exp(x)) - log(2) centered at 0
    fp = np.logaddexp(0, f_norm) - np.log(2.0)
    fp = np.maximum(fp, 1e-10)
    return fp / np.sum(fp)

def _time_to_freq(f, fft_size):
    """Convert time-domain signal to real FFT coefficients."""
    return np.fft.rfft(f, n=fft_size)

def _freq_to_time(F, fft_size):
    """Convert real FFT coefficients back to time-domain signal."""
    return np.fft.irfft(F, n=fft_size)

def _spectral_coordinate_descent(f_init, start_time, max_time=120.0):
    """
    Coordinate descent in the frequency domain.
    Each coordinate corresponds to a DFT coefficient magnitude/phase adjustment.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    fft_size = _next_pow2(2 * N - 1)
    
    best_f = f.copy()
    best_val = _compute_score(f)
    
    # Work in frequency domain
    F = _time_to_freq(f, fft_size)
    
    # Number of real FFT coefficients
    n_freq = len(F)
    
    # Precompute: for each frequency bin k, the contribution to autoconvolution
    # Autoconv in freq domain: |F|^2, so changing F[k] affects conv via |F|^2
    
    for sweep in range(200):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for k in range(1, n_freq):  # Skip DC component (k=0)
            if time.time() - start_time > max_time:
                break
            
            F_old = F[k].copy()
            
            # Objective: minimize max(autoconv) / sum^2
            # In freq domain, autoconv = irfft(|F|^2)
            # Changing F[k] changes |F[k]|^2 and cross terms with F[-k] (but we use rfft, so k and N-k are paired)
            
            # For rfft, F[k] for k in [1, N-1] are complex, and F[N/2] (if N even) is real
            # |F|^2 at bin k is |F[k]|^2 for k < N/2, and F[N/2]^2 for k = N/2
            
            def obj(scale):
                """Try scaling F[k] by 'scale' factor."""
                if scale < 0:
                    return np.inf
                F_new = F.copy()
                F_new[k] = F_old * scale
                # Ensure conjugate symmetry is maintained (rfft handles this)
                f_new = _freq_to_time(F_new, fft_size)[:N]
                f_new = np.maximum(f_new, 0.0)
                s_new = np.sum(f_new)
                if s_new < 1e-15:
                    return np.inf
                f_new = f_new / s_new
                conv = _fft_autoconv(f_new)
                return np.max(conv)
            
            # Try a few scale factors
            best_scale = 1.0
            best_obj = obj(1.0)
            for scale in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]:
                val = obj(scale)
                if val < best_obj:
                    best_obj = val
                    best_scale = scale
            
            if best_scale != 1.0:
                F[k] = F_old * best_scale
                improved = True
        
        # Convert back to time domain, project to non-negative, renormalize
        f = _freq_to_time(F, fft_size)[:N]
        f = np.maximum(f, 0.0)
        s = np.sum(f)
        if s > 1e-15:
            f = f / s
        else:
            f = best_f.copy()
        
        # Update FFT
        F = _time_to_freq(f, fft_size)
        
        current_val = _compute_score(f)
        if current_val < best_val - 1e-12:
            best_val = current_val
            best_f = f.copy()
        
        if not improved:
            break
    
    return best_f

def _time_domain_polish(f_init, start_time, max_time=120.0):
    """
    Fast time-domain coordinate descent with vectorized evaluation.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = _compute_score(f)
    
    for sweep in range(50):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > max_time:
                break
            
            old_val = f[i]
            S = np.sum(f)
            
            # Compute background conv (with f[i]=0)
            f_bg = f.copy()
            f_bg[i] = 0.0
            conv_bg = _fft_autoconv(f_bg)
            
            # Affected lags: m where m-i in [0, N-1]
            m_start = max(0, i)
            m_end = min(2 * N - 2, i + N - 1)
            js = np.arange(m_start, m_end + 1) - i
            
            # Precompute background conv values for affected lags
            conv_bg_affected = conv_bg[m_start:m_end + 1]
            
            # Precompute f[js] for off-diagonal terms
            f_js = f[js]
            mask_diag = (js == i)
            f_js_offdiag = f_js[~mask_diag]
            
            def objective(v):
                if v < 0:
                    return np.inf
                S_new = S - old_val + v
                if S_new < 1e-15:
                    return np.inf
                
                # New conv for affected lags
                new_conv = conv_bg_affected.copy()
                # Add diagonal term
                new_conv[mask_diag] += v * v
                # Add off-diagonal terms
                new_conv[~mask_diag] += 2 * v * f_js_offdiag
                
                max_conv = np.max(new_conv)
                return max_conv / (S_new * S_new)
            
            # Try a few candidate values
            candidates_v = [0.0, old_val * 0.5, old_val, old_val * 1.5, old_val * 2.0, 1.0 / N, 2.0 / N, 5.0 / N]
            best_v = old_val
            best_obj = objective(old_val)
            for v in candidates_v:
                if v < 0 or v > 1.0:
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
    
    # Generate initial candidates
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
    candidates = []
    
    # Uniform
    f = np.ones(N)
    candidates.append(f / np.sum(f))
    
    # Gaussian
    for alpha in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        f = np.exp(-alpha * x ** 2)
        candidates.append(f / np.sum(f))
        
    # Triangle
    for w in [0.1, 0.2, 0.25, 0.4, 0.5]:
        f = np.zeros(N)
        mask = np.abs(x) <= w / 2.0
        f[mask] = 1.0 - 2.0 * np.abs(x[mask]) / w
        f = np.maximum(f, 0)
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))
    
    # Raised cosine
    for beta in [0.3, 0.5, 0.7, 1.0]:
        f = np.zeros(N)
        mask = np.abs(x) <= beta / 2.0
        f[mask] = 0.5 * (1 + np.cos(2 * np.pi * x[mask] / beta))
        f = np.maximum(f, 0)
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))
    
    # Score them
    scored = []
    for f in candidates:
        score = _compute_score(f)
        scored.append((f, score))
    
    scored.sort(key=lambda x: x[1])
    best_f = scored[0][0].copy()
    best_score = scored[0][1]
    
    # Apply spectral coordinate descent to top candidates
    for f_init, score_init in scored[:5]:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        remaining = TIME_LIMIT - 5 - (time.time() - start_time)
        if remaining < 5:
            break
        refined = _spectral_coordinate_descent(f_init, start_time, remaining)
        refined_score = _compute_score(refined)
        if refined_score < best_score:
            best_f = refined
            best_score = refined_score
    
    # Polish with time-domain coordinate descent
    if time.time() - start_time < TIME_LIMIT - 10:
        remaining = TIME_LIMIT - 5 - (time.time() - start_time)
        if remaining > 5:
            polished = _time_domain_polish(best_f, start_time, remaining)
            polished_score = _compute_score(polished)
            if polished_score < best_score:
                best_f = polished
                best_score = polished_score
    
    # Final normalization
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]
