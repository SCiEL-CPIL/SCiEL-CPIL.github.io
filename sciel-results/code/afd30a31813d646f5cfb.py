"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a low-dimensional parametric family based on Gaussian bumps,
then refine with a spectral-domain coordinate descent that directly targets
peak reduction in the autoconvolution.

The key insight is to:
1. Start with a smooth parametric family (Gaussian bumps) to find a good
   initial shape.
2. Use spectral-domain optimization to directly minimize the peak of the
   autoconvolution by adjusting Fourier coefficients.
3. Apply a final polish pass with tight tolerance.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 148  # safety margin before 150s kill

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

def soft_max_weights(x, beta):
    """Compute softmax weights for the top elements, used for smooth gradient."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return np.zeros_like(x)
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    return weights

def gaussian_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from Gaussian bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        result += h * np.exp(-0.5 * ((x - c) / w) ** 2)
    return result

def parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    widths = np.abs(widths) + 0.02
    widths = np.minimum(widths, 0.5)
    heights = np.abs(heights)
    
    f = gaussian_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=10.0):
    f = parametric_to_step(params, n_components)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def spectral_refinement(h, start_time, max_time):
    """Spectral-domain coordinate descent to minimize autoconvolution peak.
    
    Uses smooth surrogate (soft-max) for better gradient information.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    n_fft = 2 * N
    
    alpha = 0.03
    no_improve_count = 0
    max_no_improve = 40
    beta = 20.0  # smoothness parameter for soft-max
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        # Compute autoconvolution via FFT
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
        
        # Use soft-max weights for smoother gradient
        weights = soft_max_weights(conv, beta)
        
        # Compute gradient using weighted combination of peak sensitivities
        # d(conv[k])/df[i] = sum over j where k-i-j=0 mod (2N-1) of f[j]
        # We use FFT to compute this efficiently
        # The sensitivity of conv at position k w.r.t. f[i] is:
        # For linear convolution: if 0 <= k-i < N, sensitivity = f[k-i]
        # We can compute the weighted sum using FFT
        
        # Compute the "sensitivity function" via circular convolution trick
        # For each i, the contribution to the weighted peak is:
        # sum_k weights[k] * d(conv[k])/df[i]
        
        # d(conv[k])/df[i] = f[k-i] if 0 <= k-i < N (for linear conv)
        # This is equivalent to: for each i, the vector of sensitivities across k is
        # a shifted version of f
        
        # Efficient computation: 
        # Let g = weights (length 2N-1), pad to length 2N
        # The gradient[i] = sum_k weights[k] * f[k-i] (with bounds)
        # This is a cross-correlation type operation
        
        # Use FFT-based approach:
        # grad[i] = sum_k weights[k] * f[k-i]
        # = (weights * f)[i] in some sense
        
        # More precisely, define extended arrays:
        # w_ext = zeros(2N), w_ext[:2N-1] = weights
        # f_ext = zeros(2N), f_ext[:N] = current_h
        # Then grad[i] = sum_k w_ext[k] * f_ext[k-i] for appropriate k
        # This is a convolution: (w_ext * f_ext)[i] but with sign flip
        
        # Actually: sum_k w[k] * f[k-i] = (w_rev * f)[i] where w_rev[k] = w[-k]
        # Let's just do it directly with FFT
        
        w_ext = np.zeros(2 * N)
        w_ext[:2*N-1] = weights
        f_ext = np.zeros(2 * N)
        f_ext[:N] = current_h
        
        # grad[i] = sum_k w_ext[k] * f_ext[k-i]
        # = correlation of w_ext and f_ext
        # = real part of ifft(conj(fft(w_ext)) * fft(f_ext))
        
        w_hat = np.fft.rfft(w_ext, n=2*N)
        f_hat2 = np.fft.rfft(f_ext, n=2*N)
        grad = np.fft.irfft(np.conj(w_hat) * f_hat2, n=2*N)[:N]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad = grad / grad_norm
        
        # Update: move in negative gradient direction
        std_h = np.std(current_h)
        h_new = current_h - alpha * grad * std_h
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        # Evaluate
        f_new_hat = np.fft.rfft(h_new, n=n_fft)
        conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.15)
            beta = min(beta * 1.1, 50.0)  # gradually increase smoothness
        else:
            alpha *= 0.9
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def final_polish(h, start_time, max_time):
    """Final high-precision polish with soft-max surrogate and adaptive step size."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    n_fft = 2 * N
    alpha = 0.01
    no_improve_count = 0
    max_no_improve = 60
    beta = 30.0  # smoothness for soft-max
    
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
        
        # Use soft-max weights for smooth gradient
        weights = soft_max_weights(conv, beta)
        
        # Compute gradient via FFT-based correlation
        w_ext = np.zeros(2 * N)
        w_ext[:2*N-1] = weights
        f_ext = np.zeros(2 * N)
        f_ext[:N] = current_h
        
        w_hat = np.fft.rfft(w_ext, n=2*N)
        f_hat2 = np.fft.rfft(f_ext, n=2*N)
        sens = np.fft.irfft(np.conj(w_hat) * f_hat2, n=2*N)[:N]
        
        # Normalize sensitivity
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        # Multiplicative update with clipping
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        f_new_hat = np.fft.rfft(h_new, n=n_fft)
        conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.015, 0.05)
            beta = min(beta * 1.05, 80.0)  # increase smoothness gradually
        else:
            alpha *= 0.92
            if alpha < 1e-9:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of Gaussian bump components
    param_budget_end = start_time + (TIME_LIMIT - 5) * 0.30
    
    for n_c in [2, 3, 4, 5]:
        if time.time() > param_budget_end:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        betas = [10.0, 20.0, 50.0]
        
        for beta in betas:
            if time.time() > param_budget_end:
                break
            
            for restart in range(4):
                if time.time() > param_budget_end:
                    break
                
                centers = rng.uniform(-0.15, 0.15, size=n_c)
                widths = rng.uniform(0.05, 0.20, size=n_c)
                heights = rng.uniform(0.5, 3.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 600, 'xatol': 1e-6, 'fatol': 1e-8}
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
    
    # Spectral refinement
    if time.time() < TIME_LIMIT - 12:
        refined_f, refined_score = spectral_refinement(
            best_f, start_time, TIME_LIMIT - 12
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Final polish
    if time.time() < TIME_LIMIT - 2:
        polished_f, polished_score = final_polish(
            best_f, start_time, TIME_LIMIT - 2
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    
    # Final normalization to ensure sum = N
    s_final = np.sum(best_f)
    if s_final > 1e-12:
        best_f = best_f * (N / s_final)
    
    # One last check: if we're close to the deadline, do a quick local search
    remaining = TIME_LIMIT - 1 - (time.time() - start_time)
    if remaining > 5:
        # Quick coordinate descent on a few random coordinates
        rng = np.random.RandomState(123)
        for _ in range(3):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            idx = rng.randint(0, N)
            for delta in [0.1, -0.1, 0.05, -0.05, 0.2, -0.2]:
                trial = best_f.copy()
                trial[idx] = max(0.0, trial[idx] + delta)
                s_t = np.sum(trial)
                if s_t < 1e-12:
                    continue
                trial = trial * (N / s_t)
                score_t = compute_score(trial)
                if score_t < best_score:
                    best_f = trial
                    best_score = score_t
                    break
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
