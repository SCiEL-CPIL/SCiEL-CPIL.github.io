"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: 
1. Use a smooth parametric family based on cosine bumps with multiple components.
2. Optimize using Nelder-Mead with a smooth max surrogate.
3. Apply local refinement using sensitivity-based peak reduction.
4. Add a final polish pass with tighter tolerances.

Key improvements over the seed:
- Better initial parameter sampling with structured diversity
- More aggressive local refinement with adaptive step sizes
- Final polish pass to squeeze out remaining improvements
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

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

def fft_autoconv(f):
    """Compute autoconvolution via FFT for speed."""
    n_conv = 2 * N - 1
    n_fft = 1 << (n_conv - 1).bit_length()
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:n_conv]
    return conv

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

def compute_sensitivity_vectorized(f, peak_indices):
    """Vectorized computation of sensitivity: for each bin j, sum of f[m-j] over peak lags m."""
    # conv[m] = sum_j f[j] * f[m-j]
    # sensitivity[j] = sum_{m in peaks} f[m-j]
    # This is equivalent to cross-correlation of f with the peak indicator
    peak_indicator = np.zeros(2 * N - 1)
    peak_indicator[peak_indices] = 1.0
    # sensitivity[j] = sum_m peak_indicator[m] * f[m-j]
    # = (peak_indicator * f)[j] in the appropriate range
    # Actually: sensitivity[j] = sum_m PI[m] * f[m-j] for valid m,j
    # This is the cross-correlation of PI with f, evaluated at lags 0..N-1
    # Use FFT: sensitivity = ifft(FFT(PI) * conj(FFT(f)))
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=n_fft)
    PI = np.fft.rfft(peak_indicator, n=n_fft)
    # Cross-correlation: ifft(conj(F) * PI) gives sum_j f[j] * PI[j+k]
    # We want sum_m PI[m] * f[m-j] = sum_k PI[k+j] * f[k] = cross_corr at lag j
    # cross_corr[j] = sum_k f[k] * PI[k+j]
    cc = np.fft.irfft(np.conj(F) * PI, n=n_fft)
    # cc[j] for j=0..N-1 is what we want (with proper indexing)
    sensitivity = cc[:N].copy()
    return sensitivity

def local_refinement(h, start_time, max_time, alpha_init=0.08, threshold_frac=0.92):
    """Local refinement using sensitivity-based peak reduction (vectorized)."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    n_iterations = 0
    
    while time.time() - start_time < max_time:
        n_iterations += 1
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_autoconv(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Vectorized sensitivity computation
        sensitivity = compute_sensitivity_vectorized(current_h, peak_indices)
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity (they contribute to peaks)
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autoconv(h_new)
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
    """Final high-precision polish pass with very small step sizes (vectorized)."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_autoconv(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Very tight threshold for polish
        threshold = max_c * 0.99
        peak_indices = np.where(conv >= threshold)[0]
        
        # Vectorized sensitivity
        sensitivity = compute_sensitivity_vectorized(current_h, peak_indices)
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autoconv(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.15)
        else:
            alpha *= 0.9
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def coordinate_descent(h, start_time, max_time):
    """Direct coordinate descent on the full 600-dim space.
    
    For each coordinate, find the optimal value that minimizes the max autoconvolution.
    Uses the fact that conv is quadratic in each coordinate.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    s = np.sum(current_h)
    current_h = current_h * (N / s)  # normalize
    
    # Precompute: conv[m] = sum_j f[j]*f[m-j]
    # For coordinate j, conv[m] depends on f[j] linearly for most m
    # d(conv[m])/df[j] = f[m-j] + f[j-m] (with appropriate bounds)
    
    n_rounds = 0
    while time.time() - start_time < max_time:
        n_rounds += 1
        improved = False
        
        for j in range(N):
            if time.time() - start_time > max_time - 1:
                break
            
            # Current contribution of f[j] to each conv[m]:
            # conv[m] = ... + f[j]*f[m-j] + f[m-j]*f[j] + ... (double counted if m-j == j)
            # Actually conv[m] = sum_k f[k]*f[m-k], so terms involving f[j]:
            # k=j: f[j]*f[m-j], k=m-j: f[m-j]*f[j]
            # If j == m-j (i.e., m=2j), it's f[j]^2
            
            # Let's compute the baseline conv without f[j]
            # conv_base[m] = conv[m] - f[j]*f[m-j] - (1 if m != 2j else 0)*f[j]*f[m-j]
            # This is getting complex. Let's use a simpler approach:
            # Try a few candidate values for f[j] and pick the best
            
            old_val = current_h[j]
            candidates = [0.0, old_val * 0.5, old_val * 0.8, old_val, old_val * 1.2, old_val * 1.5, old_val * 2.0]
            # Also try values based on sensitivity
            conv = fft_autoconv(current_h)
            max_c = np.max(conv)
            
            # Sensitivity of max conv to f[j]:
            # For m near the peak, d(conv[m])/df[j] = f[m-j] + (1 if m-j != j else 0)*f[j]
            # Approximate: use the peak lag
            peak_m = np.argmax(conv)
            dj = peak_m - j
            if 0 <= dj < N and dj != j:
                sens_j = current_h[dj] + current_h[j - dj] if 0 <= j - dj < N else current_h[dj]
            elif 0 <= dj < N:
                sens_j = 2.0 * current_h[j]
            else:
                sens_j = 0.0
            
            if sens_j > 1e-12:
                # Try reducing f[j]
                candidates.append(old_val * (1.0 - 0.1))
                candidates.append(old_val * (1.0 - 0.2))
            
            best_candidate = old_val
            best_candidate_val = best_val
            
            # Quick evaluation: only compute conv if promising
            for c in candidates:
                if c < 0 or c > old_val * 5 + 1:
                    continue
                current_h[j] = c
                # Renormalize
                s_new = np.sum(current_h)
                if s_new < 1e-12:
                    current_h[j] = old_val
                    continue
                h_temp = current_h * (N / s_new)
                conv_temp = fft_autoconv(h_temp)
                val_temp = np.max(conv_temp) / (N * N)
                if val_temp < best_candidate_val:
                    best_candidate_val = val_temp
                    best_candidate = c
            
            current_h[j] = best_candidate
            s_new = np.sum(current_h)
            current_h = current_h * (N / s_new)
            
            conv = fft_autoconv(current_h)
            new_val = np.max(conv) / (N * N)
            
            if new_val < best_val - 1e-12:
                best_val = new_val
                best_h = current_h.copy()
                improved = True
        
        if not improved and n_rounds > 3:
            break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Parametric cosine bump optimization (reduced budget)
    for n_c in [3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 50:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        betas = [5.0, 15.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 30:
                break
            
            for restart in range(6):
                if time.time() - start_time > TIME_LIMIT - 25:
                    break
                
                if restart == 0:
                    centers = np.linspace(-0.15, 0.15, n_c)
                    widths = np.full(n_c, 0.15)
                    heights = np.full(n_c, 1.0)
                elif restart == 1:
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.05, 0.25, size=n_c)
                    heights = rng.uniform(0.5, 2.0, size=n_c)
                else:
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.03, 0.3, size=n_c)
                    heights = rng.uniform(0.3, 2.5, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                    )
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_params = result.x
                except Exception:
                    pass
        
        if best_params is not None:
            f = parametric_to_step(best_params, n_c)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Phase 2: Local refinement (vectorized, fast)
    if time.time() - start_time < TIME_LIMIT - 20:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 12
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Phase 3: Coordinate descent on full space
    if time.time() - start_time < TIME_LIMIT - 10:
        cd_f, cd_score = coordinate_descent(
            best_f, start_time, TIME_LIMIT - 5
        )
        if cd_score < best_score:
            best_f = cd_f
            best_score = cd_score
    
    # Phase 4: Final polish pass (vectorized)
    if time.time() - start_time < TIME_LIMIT - 3:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 1
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
