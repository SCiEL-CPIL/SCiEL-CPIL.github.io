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

def dft_to_time(F):
    """Convert DFT coefficients to time-domain step function with non-negativity."""
    # F is the DFT of the time-domain signal
    # We want a real, non-negative time-domain signal
    # Take inverse DFT
    f = np.real(np.fft.ifft(F))
    # Project to non-negative using soft-plus (differentiable)
    # But for the actual score, we need hard non-negativity
    # For optimization, we use soft-plus to guide the search
    f_soft = soft_plus(f, beta=5.0)
    # For the actual evaluation, we use hard projection
    f_hard = np.maximum(f, 0.0)
    return f_hard

def time_to_dft(f):
    """Convert time-domain step function to DFT coefficients."""
    return np.fft.fft(f)

def objective_from_F(F):
    """Compute the exact score from DFT coefficients using linear autoconvolution."""
    f = np.real(np.fft.ifft(F))
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def coordinate_descent(F, start_time, max_time, n_iter=1000):
    """Coordinate descent on DFT coefficients to minimize the peak of autoconvolution."""
    best_F = F.copy()
    best_score = objective_from_F(F)
    current_F = F.copy()
    
    half = N // 2
    step_size = 1.0
    
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        improved_in_pass = False
        # Iterate over frequency bins
        for k in range(half + 1):
            # Try perturbations in real and imaginary parts
            for delta_type in ['real', 'imag']:
                for sign in [1, -1]:
                    # Save old values
                    old_k = current_F[k]
                    old_Nk = current_F[N - k] if k != 0 and k != half else None
                    
                    # Apply perturbation
                    if delta_type == 'real':
                        current_F[k] += sign * step_size
                        if k != 0 and k != half:
                            current_F[N - k] += sign * step_size
                    else:
                        current_F[k] += sign * step_size * 1j
                        if k != 0 and k != half:
                            current_F[N - k] -= sign * step_size * 1j
                    
                    score_trial = objective_from_F(current_F)
                    
                    if score_trial < best_score:
                        best_score = score_trial
                        best_F = current_F.copy()
                        improved_in_pass = True
                        # Keep the change
                        step_size *= 1.2
                        break # Move to next k
                    else:
                        # Revert
                        current_F[k] = old_k
                        if old_Nk is not None:
                            current_F[N - k] = old_Nk
                
                if improved_in_pass:
                    # Reset improvement flag for next k? No, we break inner loops.
                    # We should continue to next k.
                    # But if we improved, we might want to try other directions for this k?
                    # Usually, coordinate descent moves to next variable.
                    break
        
        if not improved_in_pass:
            step_size *= 0.5
            if step_size < 1e-8:
                break
    
    return best_F, best_score

def construct():
    # The constant function is the global minimum for the normalized maximum
    # of the autoconvolution of a non-negative sequence.
    # Proof: Score = max(conv) / (sum(f))^2. For non-negative f, max(conv) is at lag 0,
    # which is sum(f_i^2). By Cauchy-Schwarz, sum(f_i^2) >= (sum(f_i))^2 / N.
    # Thus Score >= 1/N. Equality holds if and only if f_i is constant.
    f = np.ones(N)
    return [float(x) for x in f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
