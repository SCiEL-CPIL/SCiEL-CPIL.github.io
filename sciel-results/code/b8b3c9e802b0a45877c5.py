import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize
import sys

N = 600
TIME_LIMIT = 145  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def cosine_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from cosine bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        mask = dist <= w / 2.0
        if np.any(mask):
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
    return result

def parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    widths = np.abs(widths) + 0.01
    widths = np.minimum(widths, 0.5)
    heights = np.abs(heights)
    
    f = cosine_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=10.0):
    f = parametric_to_step(params, n_components)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def local_refinement(h, start_time, max_time, alpha_init=0.08, threshold_frac=0.92):
    """Local refinement using sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = np.convolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Vectorized sensitivity computation
        # For each peak lag m, contribution to bin j is h[m-j]
        # We can use correlation-like structure
        # sensitivity[j] = sum_{m in peaks} h[m-j] where 0 <= m-j < N
        # This is equivalent to convolving h with a mask of peaks, but shifted.
        # Let's stick to the loop for clarity but optimize if possible.
        # Actually, for N=600, the loop is fast enough if peak_indices is small.
        # If peak_indices is large (e.g., uniform), it's O(N^2).
        
        sensitivity = np.zeros(N)
        # Optimization: if peak_indices covers a contiguous range or is large,
        # we can use FFT or matrix ops, but let's try a simple vectorized approach first.
        # sensitivity[j] = sum_{m in peaks} h[(m-j) % N] ? No, it's linear convolution indices.
        # conv[m] = sum_{j=0}^{N-1} h[j] h[m-j] (with appropriate bounds)
        # Actually np.convolve(f, f, 'full') has length 2N-1.
        # conv[m] = sum_{j} f[j] f[m-j]
        # So for a fixed m, the terms are f[j] * f[m-j].
        # We want sensitivity[j] = sum_{m in peaks} f[m-j].
        # This is the cross-correlation of f with the indicator of peaks?
        # Let g be an array of length 2N-1 where g[m] = 1 if m in peaks else 0.
        # Then sensitivity[j] = sum_{m} g[m] f[m-j].
        # This is the cross-correlation of f and g.
        # scipy.signal.correlate(f, g, mode='valid')? Or 'full'?
        # Let's just do the loop, it's safe.
        
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                sensitivity[j_min:j_max+1] += current_h[m - j_min : m - j_min + (j_max - j_min + 1)][::-1]
                # Wait, the logic in the original was:
                # for j in range(j_min, j_max + 1):
                #     sensitivity[j] += current_h[m - j]
                # Let's implement this efficiently.
                # indices_j = np.arange(j_min, j_max+1)
                # indices_h = m - indices_j
                # sensitivity[indices_j] += current_h[indices_h]
                
        # Redoing the sensitivity calc correctly and efficiently
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                js = np.arange(j_min, j_max + 1)
                hs = m - js
                sensitivity[js] += current_h[hs]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.4)
        else:
            alpha *= 0.88
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = np.convolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * 0.99
        peak_indices = np.where(conv >= threshold)[0]
        
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                js = np.arange(j_min, j_max + 1)
                hs = m - js
                sensitivity[js] += current_h[hs]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.15)
        else:
            alpha *= 0.9
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components
    for n_c in [3, 4, 5, 6]:
        if time.time() - start_time > TIME_LIMIT - 40:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta
        betas = [5.0, 10.0, 20.0, 30.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 20:
                break
            
            for restart in range(8):
                if time.time() - start_time > TIME_LIMIT - 15:
                    break
                
                if restart == 0:
                    centers = np.linspace(-0.15, 0.15, n_c)
                    widths = np.full(n_c, 0.15)
                    heights = np.full(n_c, 1.0)
                elif restart == 1:
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.05, 0.3, size=n_c)
                    heights = rng.uniform(0.5, 2.0, size=n_c)
                else:
                    centers = rng.uniform(-0.25, 0.25, size=n_c)
                    widths = rng.uniform(0.02, 0.4, size=n_c)
                    heights = rng.uniform(0.1, 3.0, size=n_c)
                
                params0 = np.concatenate([centers, widths, heights])
                
                res = minimize(objective, params0, args=(n_c, beta), method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_params = res.x
        
        if best_params is not None:
            f_candidate = parametric_to_step(best_params, n_c)
            score_candidate = compute_score(f_candidate)
            
            # Local refinement
            f_refined, score_refined = local_refinement(f_candidate, start_time, TIME_LIMIT - 25)
            if score_refined < best_score:
                best_score = score_refined
                best_f = f_refined
            
            # Polish
            f_polished, score_polished = polish_pass(f_refined, start_time, TIME_LIMIT - 5)
            if score_polished < best_score:
                best_score = score_polished
                best_f = f_polished
    
    # Final check
    if time.time() - start_time > TIME_LIMIT - 2:
        return best_f.tolist()
    
    return best_f.tolist()
