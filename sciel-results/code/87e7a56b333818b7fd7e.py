"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Discrete spectral search.
Parameterize the step function by its DFT coefficients.
Enforce non-negativity via a soft-plus projection.
Directly minimize the maximum value of the inverse DFT of the squared magnitude spectrum
using coordinate descent on the frequency domain.
This decouples the peak constraint from the time-domain shape search.

Key insight: The autoconvolution in the time domain corresponds to the squared
magnitude of the DFT in the frequency domain (up to scaling). By working in the
frequency domain, we can directly control the "energy distribution" of the
autoconvolution. The maximum of the autoconvolution is related to the peak of
the inverse DFT of |F|^2. We use coordinate descent on the frequency coefficients
to flatten this peak, while projecting back to the time domain to enforce
non-negativity.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

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

def soft_plus(x, beta=1.0):
    """Soft approximation to max(x, 0) using log(1 + exp(beta*x))/beta."""
    x = np.asarray(x, dtype=np.float64)
    # Numerically stable softplus
    return np.logaddexp(0, beta * x) / beta

def objective_from_F(F):
    """
    Compute the score from DFT coefficients.
    Uses circular autoconvolution for speed during optimization.
    f = IFFT(F).
    Autoconvolution peak is approximated by max(IFFT(|F|^2)).
    Normalization is (sum(f))^2.
    """
    # f is real if F is conjugate symmetric
    f = np.real(np.fft.ifft(F))
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    
    # Power spectrum
    power = np.abs(F) ** 2
    # Circular autoconvolution
    conv_circ = np.real(np.fft.ifft(power))
    
    max_conv = np.max(conv_circ)
    return max_conv / (s * s)

def spectral_coordinate_descent(F_init, start_time, max_time):
    """
    Coordinate descent on DFT coefficients.
    Optimizes real and imaginary parts of the first N/2+1 bins.
    Uses a fixed step size with occasional restarts/reductions.
    """
    current_F = F_init.copy()
    best_F = F_init.copy()
    best_score = objective_from_F(current_F)
    
    half = N // 2
    # Indices to optimize: 0, 1, ..., N/2
    # For k=0 and k=N/2, only real part is free (imaginary part must be 0 for real signal)
    # For 1 < k < N/2, both real and imaginary parts are free.
    
    step_sizes = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    current_step_idx = 0
    no_improve_count = 0
    patience = 20 # Reduce step size if no improvement for this many full passes
    
    while time.time() - start_time < max_time:
        improved_in_pass = False
        # Shuffle order to avoid bias? Or just iterate.
        # Iterating over bins
        for k in range(half + 1):
            if time.time() - start_time > max_time:
                break
            
            step = step_sizes[current_step_idx]
            
            # Determine degrees of freedom
            if k == 0 or k == half:
                # Only real part changes
                deltas = [(-step, 0), (step, 0)]
            else:
                # Both real and imaginary parts
                deltas = [(-step, 0), (step, 0), (0, -step), (0, step)]
            
            for dr, di in deltas:
                # Create trial F
                F_trial = current_F.copy()
                
                # Update F[k]
                F_trial[k] = F_trial[k] + (dr + 1j*di)
                
                # Update F[N-k] for conjugate symmetry
                if k != 0 and k != half:
                    # F[N-k] = conj(F[k])
                    # If F[k] += (dr + i di), then F[N-k] += (dr - i di)
                    F_trial[N-k] = F_trial[N-k] + (dr - 1j*di)
                
                score_trial = objective_from_F(F_trial)
                
                if score_trial < best_score:
                    best_score = score_trial
                    best_F = F_trial.copy()
                    current_F = F_trial.copy()
                    improved_in_pass = True
                    no_improve_count = 0
                    break # Move to next k
                else:
                    no_improve_count += 1
                    
            if no_improve_count > patience:
                # No improvement for a while, reduce step size
                no_improve_count = 0
                if current_step_idx < len(step_sizes) - 1:
                    current_step_idx += 1
                else:
                    # If we've exhausted step sizes, we might break or continue with tiny steps
                    # Let's break if we are very small
                    if step_sizes[current_step_idx] < 1e-5:
                        break
                if time.time() - start_time > max_time:
                    break
        
        if not improved_in_pass and current_step_idx >= len(step_sizes) - 1:
            break

    return best_F, best_score

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Initializations in frequency domain
    # 1. Constant (DC only)
    F1 = np.zeros(N, dtype=complex)
    F1[0] = N
    
    # 2. Cosine wave + DC
    t = np.arange(N)
    f2 = np.cos(2 * np.pi * t / N) + 1.0
    F2 = np.fft.fft(f2)
    
    # 3. Sine wave + DC
    f3 = np.sin(2 * np.pi * t / N) + 1.5
    F3 = np.fft.fft(f3)
    
    # 4. Gaussian-like decay in time
    f4 = np.exp(-np.abs(t - N/2) / 50.0)
    F4 = np.fft.fft(f4)
    
    # 5. Random positive (projected)
    rng = np.random.default_rng(42)
    f5 = rng.random(N)
    F5 = np.fft.fft(f5)
    
    initializations = [F1, F2, F3, F4, F5]
    
    for F_init in initializations:
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        # Ensure conjugate symmetry
        F_sym = F_init.copy()
        half = N // 2
        for k in range(1, half + 1):
            F_sym[N - k] = np.conj(F_sym[k])
        # Ensure DC and Nyquist are real
        F_sym[0] = F_sym[0].real
        if N % 2 == 0:
            F_sym[half] = F_sym[half].real
            
        # Run spectral coordinate descent
        F_opt, score_opt = spectral_coordinate_descent(F_sym, start_time, TIME_LIMIT - 5)
        
        # Convert back to time domain
        f_opt = np.real(np.fft.ifft(F_opt))
        # Hard projection to non-negative
        f_opt = np.maximum(f_opt, 0.0)
        
        # Compute actual score (using linear convolution for accuracy)
        actual_score = compute_score(f_opt)
        
        if actual_score < best_score:
            best_score = actual_score
            best_f = f_opt
    
    # Time-domain refinement
    if time.time() - start_time < TIME_LIMIT - 10:
        h = best_f.copy()
        # Normalize to have sum N
        s = np.sum(h)
        if s > 1e-12:
            h = h * (N / s)
            
        alpha = 0.05
        for _ in range(100):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            s = np.sum(h)
            if s < 1e-12:
                break
            h_norm = h * (N / s)
            
            conv = np.convolve(h_norm, h_norm, mode='full')
            max_c = np.max(conv)
            current_val = max_c / (N * N)
            
            # Sensitivity: gradient of max(conv) wrt h
            # max is at some lag m. conv[m] = sum_j h[j] h[m-j]
            # d(conv[m])/d(h[i]) = h[m-i] + h[i-m] (with appropriate bounds)
            # We approximate sensitivity by summing contributions from top peaks
            
            threshold = max_c * 0.95
            peak_indices = np.where(conv >= threshold)[0]
            
            sensitivity = np.zeros(N)
            for m in peak_indices:
                # m ranges from 0 to 2N-2
                # h[j] contributes to conv[m] if j is in [0, N-1] and m-j is in [0, N-1]
                # j_min = max(0, m - (N-1))
                # j_max = min(N-1, m)
                j_min = max(0, m - (N - 1))
                j_max = min(N - 1, m)
                if j_min <= j_max:
                    # h[j] * h[m-j]
                    # derivative wrt h[j] is h[m-j]
                    # Note: m-j is the index in the second h
                    idx1 = np.arange(j_min, j_max + 1)
                    idx2 = m - idx1
                    sensitivity[idx1] += h_norm[idx2]
            
            sens_mean = np.mean(sensitivity)
            sens_std = np.std(sensitivity) + 1e-12
            sens_norm = (sensitivity - sens_mean) / sens_std
            
            h_new = h * (1.0 - alpha * sens_norm)
            h_new = np.maximum(h_new, 0.0)
            
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                break
            h_new = h_new * (N / s_new)
            
            conv_new = np.convolve(h_new, h_new, mode='full')
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < current_val:
                h = h_new
            else:
                alpha *= 0.9
                if alpha < 1e-5:
                    break
        
        h = np.maximum(h, 0.0)
        score_h = compute_score(h)
        if score_h < best_score:
            best_score = score_h
            best_f = h
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
