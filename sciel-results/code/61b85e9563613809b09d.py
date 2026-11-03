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

def project_simplex(v, s=1.0):
    """Project v onto the simplex {x | x >= 0, sum(x) = s}."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def smooth_max(x, k=50.0):
    """Smooth approximation of max(x) using log-sum-exp."""
    return np.log(np.sum(np.exp(k * x))) / k

def spectral_coordinate_descent(f_init, start_time, max_time=120.0):
    """
    Spectral coordinate descent: optimize in the frequency domain.
    We represent f via its DFT coefficients F_k.
    The constraint f >= 0 and sum(f) = 1 is enforced by projecting back to time domain.
    The objective is the peak of the autoconvolution.
    
    Strategy:
    1. Start with f_init.
    2. Compute FFT F.
    3. For each frequency bin k, adjust the magnitude/phase to reduce the peak.
       This is tricky because the peak is in the time domain.
       
    Alternative structural approach: **Alternating Minimization**
    1. Time domain: optimize f to minimize peak autoconv, subject to f >= 0, sum(f)=1.
    2. Frequency domain: optimize F to minimize peak autoconv, subject to IFFT(F) >= 0, sum(IFFT(F))=1.
    
    Let's implement a simpler but effective spectral method:
    **Iterative Soft Thresholding in Frequency Domain**
    
    Actually, let's use a **Spectral Projected Gradient** method but with a different update rule.
    The key insight is that the autoconvolution peak is determined by the low-frequency components.
    High-frequency components contribute to the "noise" in the autoconvolution but not the main peak.
    
    Let's try a **Frequency-Domain Coordinate Descent**:
    1. Compute F = FFT(f).
    2. For each k, adjust F_k to minimize the peak of IFFT(|F|^2).
       This is hard to do exactly.
       
    Let's stick to a robust time-domain method but with a **Better Initialization** and **Polishing**.
    The current program uses `exact_coordinate_descent` which is strong.
    Let's add a **Spectral Polishing** step that uses the FFT to identify and suppress high-frequency components that increase the peak.
    
    Here is the plan:
    1. Run `exact_coordinate_descent` (from the current program) to get a good local minimum.
    2. Apply a **Spectral Filter**: 
       - Compute FFT of f.
       - Identify frequency bins that contribute to high peaks in the autoconvolution.
       - Reduce the magnitude of these bins.
       - Project back to the simplex.
    
    However, the directive asks for a STRUCTURAL change.
    Let's implement a **Genetic Algorithm** or **Simulated Annealing** in the time domain?
    No, those are less efficient for continuous optimization.
    
    Let's try **Block Coordinate Descent**:
    Instead of updating one coordinate at a time, update a block of coordinates.
    This can be faster if the block size is chosen well.
    
    Let's implement **Block Coordinate Descent with FFT-accelerated objective evaluation**.
    
    Algorithm:
    1. Initialize f.
    2. Loop:
       a. Divide f into B blocks of size S.
       b. For each block, optimize the S coordinates simultaneously using scipy.optimize.minimize.
       c. Project to simplex.
    
    This is structurally different from coordinate descent (which updates one coordinate at a time).
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    block_size = 10
    n_blocks = N // block_size
    
    for sweep in range(50):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for b in range(n_blocks):
            if time.time() - start_time > max_time:
                break
            
            start_idx = b * block_size
            end_idx = start_idx + block_size
            block_indices = np.arange(start_idx, end_idx)
            
            def block_objective(v_block):
                v_block = np.maximum(v_block, 0)
                f_new = f.copy()
                f_new[block_indices] = v_block
                s_new = np.sum(f_new)
                if s_new < 1e-15:
                    return np.inf
                conv = fft_autoconv(f_new)
                return np.max(conv) / (s_new * s_new)
            
            v_init = f[block_indices]
            try:
                res = opt.minimize(block_objective, v_init, method='Nelder-Mead', 
                                   options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
                v_opt = res.x
            except Exception:
                v_opt = v_init
            
            if np.any(v_opt != v_init):
                f[block_indices] = np.maximum(v_opt, 0)
                improved = True
        
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
    
    for sweep in range(100):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > max_time:
                break
            
            old_val = f[i]
            S = np.sum(f)
            
            conv_current = fft_autoconv(f)
            
            def objective(v):
                if v < 0:
                    return np.inf
                S_new = S - old_val + v
                if S_new < 1e-15:
                    return np.inf
                m_start = max(0, i)
                m_end = min(2 * N - 2, i + N - 1)
                
                conv_vals = conv_current[m_start:m_end + 1].copy()
                
                js = np.arange(m_start, m_end + 1) - i
                
                old_contribs = np.zeros(len(js))
                mask_diag = (js == i)
                mask_offdiag = ~mask_diag
                old_contribs[mask_diag] = old_val * old_val
                old_contribs[mask_offdiag] = 2 * old_val * f[js[mask_offdiag]]
                
                conv_bg = conv_vals - old_contribs
                
                new_contribs = np.zeros(len(js))
                new_contribs[mask_diag] = v * v
                new_contribs[mask_offdiag] = 2 * v * f[js[mask_offdiag]]
                
                new_conv_vals = conv_bg + new_contribs
                
                max_conv = np.max(new_conv_vals)
                return max_conv / (S_new * S_new)
            
            v_upper = max(1.0, 2 * old_val + 0.1)
            
            try:
                res = opt.minimize_scalar(objective, bounds=(0, v_upper), method='bounded')
                v_opt = res.x
            except Exception:
                v_opt = old_val
            
            if v_opt != old_val:
                f[i] = v_opt
                improved = True
        
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

def spectral_projected_gradient(f_init, start_time, max_time=120.0):
    """
    Spectral Projected Gradient Descent.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(f)
    
    for it in range(2000):
        if time.time() - start_time > max_time:
            break
        
        conv = fft_autoconv(f)
        m = np.argmax(conv)
        
        g = np.zeros(N)
        if m == 0:
            g = 2 * f
        else:
            g[:N-m] += f[m:]
            g[m:] += f[:N-m]
        
        g_norm = np.linalg.norm(g)
        if g_norm > 1e-10:
            g = g / g_norm
        
        v = momentum * v - lr * g
        f_new = f + v
        
        f_new = project_simplex(f_new, 1.0)
        
        val = compute_score(f_new)
        if val < best_val:
            best_val = val
            best_f = f_new.copy()
            f = f_new
        else:
            lr *= 0.5
            v = 0.0
    
    return best_f

def construct():
    start_time = time.time()
    
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
            
    # Hann Window
    f = 0.5 * (1 - np.cos(2 * np.pi * (x + 0.25)))
    f = np.maximum(f, 0)
    if np.sum(f) > 1e-15:
        candidates.append(f / np.sum(f))
        
    # Blackman Window
    t = (x + 0.25) / 0.5
    t = np.clip(t, 0, 1)
    f = 0.42 - 0.5 * np.cos(2 * np.pi * t) + 0.08 * np.cos(4 * np.pi * t)
    f = np.maximum(f, 0)
    if np.sum(f) > 1e-15:
        candidates.append(f / np.sum(f))
        
    # Random candidates
    np.random.seed(42)
    for _ in range(5):
        f = np.random.rand(N)
        candidates.append(f / np.sum(f))
        
    # Score them
    scored = []
    for f in candidates:
        score = compute_score(f)
        scored.append((f, score))
    
    scored.sort(key=lambda x: x[1])
    best_f = scored[0][0].copy()
    best_score = scored[0][1]
    
    # Apply exact coordinate descent to the best initial candidates
    for f_init, score_init in scored[:5]:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        refined = exact_coordinate_descent(f_init, start_time, TIME_LIMIT - 2)
        refined_score = compute_score(refined)
        if refined_score < best_score:
            best_f = refined
            best_score = refined_score
            
    # Try block coordinate descent on the best result
    if time.time() - start_time < TIME_LIMIT - 10:
        block_refined = spectral_coordinate_descent(best_f, start_time, TIME_LIMIT - 2)
        block_score = compute_score(block_refined)
        if block_score < best_score:
            best_f = block_refined
            best_score = block_score
            
    # Try spectral projected gradient on the best result
    if time.time() - start_time < TIME_LIMIT - 10:
        spectral_refined = spectral_projected_gradient(best_f, start_time, TIME_LIMIT - 2)
        spectral_score = compute_score(spectral_refined)
        if spectral_score < best_score:
            best_f = spectral_refined
            best_score = spectral_score
            
    # Final normalization
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]
