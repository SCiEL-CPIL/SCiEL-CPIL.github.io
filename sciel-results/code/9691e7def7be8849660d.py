"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Cosine-bump parametric initialization via Nelder-Mead (gives good starting shape).
2. Spectral (frequency-domain) optimization with log-sum-exp smooth surrogate:
   - Work in the FFT domain; peak of autoconvolution = max of inverse FFT of |F|^2.
   - Gradient of smooth_max w.r.t. f is 2 * corr(weights, f), computed via FFT.
   - Use projected gradient descent / spectral updates to reduce the peak.
3. Time-domain coordinate refinement for fine-tuning.
4. Final polish pass.
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
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for autoconvolution
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2*N-1]
    return np.max(conv) / (s * s)

def fft_autoconv(f):
    """Compute autoconvolution using FFT."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2*N-1]
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
    conv = fft_autoconv(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def spectral_optimization(f_init, start_time, max_time, beta=20.0, lr=0.001, decay=0.999):
    """Spectral (frequency-domain) optimization using projected gradient descent.
    
    The autoconvolution is the inverse FFT of |F|^2. We want to minimize the peak
    of this autoconvolution. Using a smooth max surrogate, the gradient w.r.t. f
    can be computed efficiently via FFT.
    
    The key insight: conv = IFFT(|FFT(f)|^2), so d(conv)/df = 2 * IFFT(FFT(f) * conj(weights))
    where weights = softmax(beta * conv) / sum(softmax(beta * conv))
    """
    best_f = f_init.copy()
    best_val = compute_score(f_init)
    current_f = f_init.copy()
    
    s = np.sum(current_f)
    if s > 1e-12:
        current_f = current_f * (N / s)
    
    current_val = best_val
    lr_current = lr
    no_improve = 0
    max_no_improve = 200
    
    while time.time() - start_time < max_time and no_improve < max_no_improve:
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)
        
        conv = fft_autoconv(current_f)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Compute weights using softmax
        shifted = conv / max_c - 1.0
        weights = np.exp(beta * shifted)
        weights = weights / np.sum(weights)
        
        # Gradient of smooth_max w.r.t. f
        # conv = IFFT(|FFT(f)|^2)
        # d(conv[k])/d(f[j]) involves correlation
        # Using FFT: grad_f = 2 * IFFT(FFT(f) * conj(FFT(weights)))
        n_fft = 1
        while n_fft < 2 * N - 1:
            n_fft *= 2
        
        F = np.fft.rfft(current_f, n=n_fft)
        W = np.fft.rfft(weights, n=n_fft)
        
        # Gradient computation
        grad = np.fft.irfft(F * np.conj(W), n=n_fft)[:N]
        grad = 2.0 * grad
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        grad = grad / grad_norm
        
        # Projected gradient step
        f_new = current_f - lr_current * grad
        f_new = np.maximum(f_new, 0.0)
        
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            lr_current *= decay
            continue
        f_new = f_new * (N / s_new)
        
        conv_new = fft_autoconv(f_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_f = f_new
            lr_current = min(lr_current * 1.05, 0.05)
            no_improve = 0
        else:
            lr_current *= decay
            if lr_current < 1e-10:
                break
    
    return best_f, best_val

def compute_sensitivity_fft(h, peak_indices):
    """Compute sensitivity using FFT correlation (vectorized)."""
    sensitivity = np.zeros(N)
    if len(peak_indices) == 0:
        return sensitivity
    
    # Build indicator of peak lags
    peak_ind = np.zeros(2*N - 1)
    peak_ind[peak_indices] = 1.0
    
    # Use FFT for cross-correlation
    n_fft = 1
    while n_fft < 4*N:
        n_fft *= 2
    
    h_padded = np.zeros(2*N - 1)
    h_padded[:N] = h
    
    H = np.fft.rfft(h_padded, n=n_fft)
    P = np.fft.rfft(peak_ind, n=n_fft)
    
    # Cross-correlation via FFT
    corr = np.fft.irfft(H * np.conj(P), n=n_fft)
    
    # Extract the relevant part
    offsets = np.arange(N)
    for m in peak_indices:
        j_min = max(0, m - (N - 1))
        j_max = min(N - 1, m)
        if j_min <= j_max:
            sensitivity[j_min:j_max+1] += h[m - np.arange(j_min, j_max+1)]
    
    return sensitivity

def local_refinement(h, start_time, max_time, alpha_init=0.08, threshold_frac=0.92):
    """Local refinement using FFT-accelerated sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < max_time and no_improve_count < max_no_improve:
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
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Adaptive threshold: start broad, narrow in
        if no_improve_count < 10:
            thr = max_c * threshold_frac
        elif no_improve_count < 30:
            thr = max_c * 0.97
        else:
            thr = max_c * 0.995
        
        peak_indices = np.where(conv >= thr)[0]
        
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])
        
        sensitivity = compute_sensitivity_fft(current_h, peak_indices)
        
        # Only modify bins that contribute to peaks
        sens_max = np.max(sensitivity)
        if sens_max < 1e-15:
            break
        
        # Normalize sensitivity to [0, 1] range for better control
        sens_norm = sensitivity / (sens_max + 1e-15)
        
        # Apply reduction only where sensitivity is significant
        mask = sens_norm > 0.1
        h_new = current_h.copy()
        h_new[mask] = current_h[mask] * (1.0 - alpha * sens_norm[mask])
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autoconv(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.5)
            no_improve_count = 0
        else:
            alpha *= 0.9
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def discrete_annealing(h, start_time, max_time):
    """Block-based annealing: modify small blocks of bins to minimize peak autoconvolution."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    current_val = best_val
    
    s = np.sum(current_h)
    if s > 1e-12:
        current_h = current_h * (N / s)
    
    T = 0.3
    cooling = 0.997
    rng = np.random.RandomState(42)
    block_sizes = [1, 2, 3, 5]
    
    while time.time() - start_time < max_time and T > 0.001:
        # Choose block size
        bs = block_sizes[rng.randint(len(block_sizes))]
        
        # Choose starting position
        j = rng.randint(N - bs + 1)
        
        # Generate perturbation for the block
        old_block = current_h[j:j+bs].copy()
        scale = 0.15 * np.maximum(np.mean(old_block), 0.1)
        perturbation = rng.normal(0, scale, size=bs)
        new_block = np.maximum(0.0, old_block + perturbation)
        
        if np.allclose(new_block, old_block):
            continue
        
        current_h[j:j+bs] = new_block
        s_new = np.sum(current_h)
        if s_new < 1e-12:
            current_h[j:j+bs] = old_block
            continue
        current_h = current_h * (N / s_new)
        
        conv_new = fft_autoconv(current_h)
        val_new = np.max(conv_new) / (N * N)
        
        accept = val_new < current_val
        if not accept and T > 0.01:
            accept = rng.random() < np.exp(-(val_new - current_val) / T)
        
        if accept:
            current_val = val_new
            if val_new < best_val:
                best_val = val_new
                best_h = current_h.copy()
        else:
            current_h[j:j+bs] = old_block
            # Renormalize after rejection
            s_rej = np.sum(current_h)
            if s_rej > 1e-12:
                current_h = current_h * (N / s_rej)
        
        T *= cooling
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass targeting the single dominant peak."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.01
    no_improve = 0
    max_no_improve = 100
    
    while time.time() - start_time < max_time and no_improve < max_no_improve:
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
            no_improve = 0
        else:
            no_improve += 1
        
        # Focus on the single dominant peak and nearby peaks
        peak_idx = np.argmax(conv)
        # Include peaks within 5% of max
        thr = max_c * 0.95
        peak_indices = np.where(conv >= thr)[0]
        
        sensitivity = compute_sensitivity_fft(current_h, peak_indices)
        
        sens_max = np.max(sensitivity)
        if sens_max < 1e-15:
            break
        
        # Gentle, targeted reduction
        sens_norm = sensitivity / (sens_max + 1e-15)
        
        # Only modify bins with significant contribution
        mask = sens_norm > 0.2
        if not np.any(mask):
            break
            
        h_new = current_h.copy()
        h_new[mask] = current_h[mask] * (1.0 - alpha * sens_norm[mask])
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autoconv(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.1)
            no_improve = 0
        else:
            alpha *= 0.92
            if alpha < 1e-12:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    for n_c in [3, 4, 5, 6]:
        if time.time() - start_time > TIME_LIMIT - 50:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        betas = [5.0, 10.0, 20.0]
        
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
    
    # Allocate time dynamically based on remaining budget
    remaining = TIME_LIMIT - (time.time() - start_time)
    
    if remaining > 20:
        # Spectral optimization first - this is the key new component
        spectral_time = min(remaining * 0.4, 40)
        spectral_f, spectral_score = spectral_optimization(
            best_f, time.time(), spectral_time, beta=20.0, lr=0.001
        )
        if spectral_score < best_score:
            best_f = spectral_f
            best_score = spectral_score
    
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        refine_time = min(remaining * 0.4, 35)
        refined_f, refined_score = local_refinement(
            best_f, time.time(), refine_time
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 10:
        anneal_time = min(remaining * 0.4, 30)
        annealed_f, annealed_score = discrete_annealing(
            best_f, time.time(), anneal_time
        )
        if annealed_score < best_score:
            best_f = annealed_f
            best_score = annealed_score
    
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 5:
        polish_time = min(remaining * 0.8, 20)
        polished_f, polished_score = polish_pass(
            best_f, time.time(), polish_time
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
