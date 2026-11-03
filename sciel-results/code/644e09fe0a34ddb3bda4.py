"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Parametric initialization with cosine bumps (smooth family)
2. FFT-based local refinement using smooth peak masks and backtracking line search
3. Haar wavelet coordinate descent to exploit piecewise-constant structure
   and escape local minima of time-domain methods
4. Final high-precision polish

The key insight is to combine the smooth sensitivity-based gradient approach
(FFT-accelerated) with a greedy coordinate descent in the Haar wavelet basis.
The Haar basis is natural for piecewise-constant functions and the orthogonality
allows us to make independent adjustments that reduce peak autoconvolution
without immediately undoing previous improvements.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 146  # safety margin before 150s kill

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

def fft_convolve(f):
    """Fast autoconvolution via FFT."""
    n_fft = 2 * N
    f_hat = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
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

def local_refinement(h, start_time, max_time):
    """Local refinement using FFT-based sensitivity peak reduction with adaptive
    threshold, smooth peak mask, and backtracking line search."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.05
    n_fft = 2 * N
    
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        f_hat = np.fft.rfft(current_h, n=n_fft)
        conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            break
        
        # Adaptive threshold: start very sharp, relax if stuck
        if no_improve_count < 5:
            threshold_frac = 0.999
        elif no_improve_count < 10:
            threshold_frac = 0.995
        elif no_improve_count < 15:
            threshold_frac = 0.99
        else:
            threshold_frac = 0.98
        
        threshold = max_c * threshold_frac
        lower_thresh = max_c * (threshold_frac - 0.01)
        peak_mask = np.clip((conv - lower_thresh) / (threshold - lower_thresh + 1e-15), 0, 1)
        # Sharpen the mask to focus on the very top
        peak_mask = peak_mask ** 3
        
        pm_hat = np.fft.rfft(peak_mask, n=n_fft)
        ch_hat = np.fft.rfft(current_h, n=n_fft)
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
        
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        best_alpha = alpha
        best_h_candidate = None
        best_val_candidate = current_val
        
        for trial_alpha in [alpha, alpha * 0.5, alpha * 2.0, alpha * 0.25, alpha * 4.0]:
            h_new = current_h * (1.0 - trial_alpha * sens_norm)
            h_new = np.maximum(h_new, 0.0)
            
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                continue
            h_new = h_new * (N / s_new)
            
            f_new_hat = np.fft.rfft(h_new, n=n_fft)
            conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < best_val_candidate:
                best_val_candidate = val_new
                best_alpha = trial_alpha
                best_h_candidate = h_new
        
        if best_h_candidate is not None and best_val_candidate < current_val:
            current_h = best_h_candidate
            if best_alpha > alpha:
                alpha = min(alpha * 1.15, 0.25)
            else:
                alpha = max(alpha * 0.9, 0.01)
        else:
            alpha *= 0.7
            if alpha < 1e-6:
                break
    
    return best_h, best_val

def haar_wavelet_refinement(h, start_time, max_time):
    """Greedy coordinate descent in Haar wavelet basis to exploit piecewise-constant
    structure and escape local minima of time-domain gradient methods.
    
    Uses structured block moves inspired by Haar wavelets: at each step,
    pick a block size and position, then adjust the first half up and second
    half down (or vice versa) to reduce peak autoconvolution.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    n_fft = 2 * N
    
    alpha = 0.05
    no_improve = 0
    max_no_improve = 40
    
    # Precompute some block sizes for structured moves
    block_sizes = [2, 4, 8, 16, 32, 64, 128]
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_convolve(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > max_no_improve:
            break
        
        # Pick a random block size and position for structured update
        bs = block_sizes[np.random.randint(len(block_sizes))]
        if bs >= N:
            bs = N // 2
        
        # Pick a random starting position aligned to block size
        n_positions = N - 2 * bs + 1
        if n_positions <= 0:
            continue
        start = np.random.randint(0, n_positions + 1, size=1)[0]
        start = (start // bs) * bs  # Align to block boundary
        
        if start + 2 * bs > N:
            continue
        
        # First half and second half of the block
        half1 = slice(start, start + bs)
        half2 = slice(start + bs, start + 2 * bs)
        
        # Try adjusting: increase first half, decrease second half (or vice versa)
        best_delta = 0.0
        best_candidate = None
        best_val_candidate = current_val
        
        for delta in [alpha * 0.5, alpha, alpha * 2.0, -alpha * 0.5, -alpha, -alpha * 2.0]:
            h_new = current_h.copy()
            h_new[half1] += delta
            h_new[half2] -= delta
            h_new = np.maximum(h_new, 0.0)
            
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                continue
            h_new = h_new * (N / s_new)
            
            conv_new = fft_convolve(h_new)
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < best_val_candidate:
                best_val_candidate = val_new
                best_delta = delta
                best_candidate = h_new
        
        if best_candidate is not None and best_val_candidate < current_val:
            current_h = best_candidate
            alpha = min(alpha * 1.08, 0.4)
        else:
            alpha *= 0.85
            if alpha < 1e-7:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Parametric initialization with cosine bumps
    # Reserve time: 20% for parametric, 40% for local refinement, 25% for wavelet, 15% for polish
    param_budget_end = start_time + (TIME_LIMIT - 5) * 0.20
    
    for n_c in [2, 3, 4]:
        if time.time() > param_budget_end:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Use fewer restarts and iterations for speed
        for restart in range(2):
            if time.time() > param_budget_end:
                break
            
            centers = rng.uniform(-0.15, 0.15, size=n_c)
            widths = rng.uniform(0.08, 0.25, size=n_c)
            heights = rng.uniform(0.5, 2.0, size=n_c)
            
            x0 = np.concatenate([centers, widths, heights])
            
            try:
                result = minimize(
                    objective, x0, args=(n_c, 15.0),
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-5, 'fatol': 1e-6}
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
    
    # Phase 2: FFT-based local refinement
    if time.time() < TIME_LIMIT - 45:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 45
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Phase 3: Haar wavelet structured coordinate descent
    if time.time() < TIME_LIMIT - 18:
        wavelet_f, wavelet_score = haar_wavelet_refinement(
            best_f, start_time, TIME_LIMIT - 18
        )
        if wavelet_score < best_score:
            best_f = wavelet_f
            best_score = wavelet_score
    
    # Phase 4: Final high-precision polish with adaptive peak targeting
    if time.time() < TIME_LIMIT - 2:
        best_f = best_f.copy()
        alpha_small = 0.003
        n_fft = 2 * N
        no_improve_polish = 0
        max_no_improve_polish = 50
        
        for _ in range(500):
            if time.time() > TIME_LIMIT - 2:
                break
            s = np.sum(best_f)
            if s < 1e-12:
                break
            best_f = best_f * (N / s)
            f_hat = np.fft.rfft(best_f, n=n_fft)
            conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
            max_c = np.max(conv)
            current_val = max_c / (N * N)
            
            # Adaptive threshold: start very sharp, relax if stuck
            if no_improve_polish < 5:
                threshold_frac = 0.9999
            elif no_improve_polish < 15:
                threshold_frac = 0.999
            elif no_improve_polish < 30:
                threshold_frac = 0.998
            else:
                threshold_frac = 0.997
            
            threshold = max_c * threshold_frac
            lower_thresh = max_c * (threshold_frac - 0.002)
            peak_mask = np.clip((conv - lower_thresh) / (threshold - lower_thresh + 1e-15), 0, 1)
            peak_mask = peak_mask ** 3  # Sharpen more
            
            pm_hat = np.fft.rfft(peak_mask, n=n_fft)
            ch_hat = np.fft.rfft(best_f, n=n_fft)
            sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
            
            sens_mean = np.mean(sens)
            sens_std = np.std(sens) + 1e-12
            sens_norm = (sens - sens_mean) / sens_std
            
            best_alpha = alpha_small
            best_candidate = None
            best_val = current_val
            
            for trial_alpha in [alpha_small, alpha_small * 0.5, alpha_small * 2.0, alpha_small * 4.0]:
                h_new = best_f * (1.0 - trial_alpha * sens_norm)
                h_new = np.maximum(h_new, 0.0)
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    continue
                h_new = h_new * (N / s_new)
                
                f_new_hat = np.fft.rfft(h_new, n=n_fft)
                conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
                val_new = np.max(conv_new) / (N * N)
                
                if val_new < best_val:
                    best_val = val_new
                    best_alpha = trial_alpha
                    best_candidate = h_new
            
            if best_candidate is not None and best_val < current_val:
                best_f = best_candidate
                alpha_small = min(alpha_small * 1.05, 0.015)
                no_improve_polish = 0
            else:
                alpha_small *= 0.8
                no_improve_polish += 1
                if alpha_small < 1e-8:
                    break
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
