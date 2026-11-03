import numpy as np
import time
import scipy.optimize as opt

N = 600
TIME_LIMIT = 140

def fft_autoconv(f):
    """Compute full autoconvolution via FFT."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    return conv[:size]

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_autoconv(f)
    return np.max(conv) / (s * s)

def exact_coordinate_descent(f_init, start_time, max_time=120.0):
    """
    Coordinate descent where each coordinate is optimized exactly
    by minimizing the normalized max autoconvolution over that coordinate.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s  # Normalize
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    # Precompute indices for efficiency?
    # For each i, the affected lags are m in [i, i+N-1] intersect [0, 2N-2]
    
    for sweep in range(100):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > max_time:
                break
            
            old_val = f[i]
            S = np.sum(f)
            
            # Define the objective function for v = f[i]
            # h(v) = max_m (C_m(v)) / (S - old_val + v)^2
            # C_m(v) = C_m_bg + delta_m(v)
            
            # Calculate C_m_bg (autoconvolution with f[i] = 0)
            # We can compute this by taking the current autoconv and subtracting the contribution of f[i]
            conv_current = fft_autoconv(f)
            
            # Contribution of f[i] to conv_current:
            # For each m, if 0 <= m-i < N:
            #   if m-i == i: term is f[i]^2
            #   else: term is 2 * f[i] * f[m-i]
            #   else: 0
            
            # We need to find v > 0 minimizing h(v).
            # Let's use scipy.optimize.minimize_scalar
            
            def objective(v):
                if v < 0:
                    return np.inf
                S_new = S - old_val + v
                if S_new < 1e-15:
                    return np.inf
                # Compute new conv values for affected lags
                # Affected lags m are those where m-i is in [0, N-1]
                # m ranges from i to i+N-1
                m_start = max(0, i)
                m_end = min(2 * N - 2, i + N - 1)
                
                # Get current conv values for these lags
                conv_vals = conv_current[m_start:m_end + 1].copy()
                
                # Subtract old contribution
                # For m in [m_start, m_end]:
                # j = m - i
                # if j == i: old_contrib = old_val^2
                # else: old_contrib = 2 * old_val * f[j]
                
                js = np.arange(m_start, m_end + 1) - i
                # js are in [0, N-1] by construction of m_start/m_end
                
                old_contribs = np.zeros(len(js))
                mask_diag = (js == i)
                mask_offdiag = ~mask_diag
                old_contribs[mask_diag] = old_val * old_val
                old_contribs[mask_offdiag] = 2 * old_val * f[js[mask_offdiag]]
                
                conv_bg = conv_vals - old_contribs
                
                # Add new contribution
                new_contribs = np.zeros(len(js))
                new_contribs[mask_diag] = v * v
                new_contribs[mask_offdiag] = 2 * v * f[js[mask_offdiag]]
                
                new_conv_vals = conv_bg + new_contribs
                
                max_conv = np.max(new_conv_vals)
                return max_conv / (S_new * S_new)
            
            # We need a good upper bound for v.
            # Since sum is 1, v cannot be larger than 1 practically.
            # But to be safe, let's use a bound based on current max.
            v_upper = max(1.0, 2 * old_val + 0.1)
            
            try:
                res = opt.minimize_scalar(objective, bounds=(0, v_upper), method='bounded')
                v_opt = res.x
            except Exception:
                v_opt = old_val
            
            if v_opt != old_val:
                f[i] = v_opt
                improved = True
        
        # Renormalize to maintain sum=1 for next sweep?
        # The objective function handled normalization internally, so f is not strictly normalized after update.
        # However, the problem is homogeneous, so scaling doesn't change the score.
        # But for numerical stability and consistency, let's renormalize.
        s = np.sum(f)
        if s > 1e-15:
            f = f / s
        
        current_val = compute_score(f)
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

    # Score them
    scored = []
    for f in candidates:
        score = compute_score(f)
        scored.append((f, score))
    
    scored.sort(key=lambda x: x[1])
    best_f = scored[0][0].copy()
    best_score = scored[0][1]
    
    # Apply exact coordinate descent to the best initial candidate
    # We can also try it on the second best to see if it escapes local minima
    for f_init, score_init in scored[:3]:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        refined = exact_coordinate_descent(f_init, start_time, TIME_LIMIT - 2)
        refined_score = compute_score(refined)
        if refined_score < best_score:
            best_f = refined
            best_score = refined_score
            
    # Final normalization
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]
