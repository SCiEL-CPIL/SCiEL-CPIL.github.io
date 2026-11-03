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
    # Use FFT for faster convolution calculation
    # Linear convolution of two length N signals requires FFT size >= 2N-1
    # We use 2*N for simplicity and safety
    f_fft = np.fft.fft(f, n=2*N)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))
    # The linear convolution is contained in conv[0:2*N-1]
    # But ifft of size 2*N gives circular convolution.
    # Since we zero-padded to 2*N, the linear convolution is correctly represented in the first 2*N-1 elements?
    # Actually, if we pad to M >= 2N-1, the first 2N-1 elements of the IFFT result are the linear convolution.
    # Here M = 2N, which is >= 2N-1.
    # So conv[0:2*N-1] is the linear autoconvolution.
    max_val = np.max(conv[:2*N-1])
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return max_val / (s * s)

def objective_from_F(F):
    """Compute the exact score from DFT coefficients using FFT-based autoconvolution."""
    f = np.real(np.fft.ifft(F))
    f = np.maximum(f, 0.0)
    return compute_score(f)

def coordinate_descent(F, start_time, max_time, n_iter=2000):
    """Coordinate descent on DFT coefficients to minimize the peak of autoconvolution."""
    best_F = F.copy()
    best_score = objective_from_F(F)
    current_F = F.copy()
    
    half = N // 2
    step_size = 1.0
    no_improve_count = 0
    
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        improved_in_pass = False
        # Randomize the order of frequency bins to avoid cyclic behavior
        # But for coordinate descent, fixed order is standard. 
        # However, random restarts of the order might help.
        # Let's stick to fixed order for now but use adaptive step size.
        
        for k in range(half + 1):
            # We optimize the magnitude and phase separately? 
            # Or just real and imaginary parts?
            # Real and Imag parts are sufficient to span the space.
            
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
                        no_improve_count = 0
                        # Keep the change
                        # Increase step size slightly if we improved
                        step_size *= 1.1
                        break # Move to next k
                    else:
                        # Revert
                        current_F[k] = old_k
                        if old_Nk is not None:
                            current_F[N - k] = old_Nk
                
                if improved_in_pass:
                    break
        
        if not improved_in_pass:
            no_improve_count += 1
            step_size *= 0.7
            if step_size < 1e-6:
                # If step size is too small, we might be stuck.
                # Try a larger random jump?
                # For now, break and rely on restarts.
                break
            if no_improve_count > 10:
                break
    
    return best_F, best_score

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Use random restarts with box-filter initializations
    # A box filter in frequency domain corresponds to a sinc-like function in time.
    # Clipping the sinc to non-negative might yield a flatter autoconvolution.
    
    n_restarts = 100 # Increased restarts since each is faster
    half = N // 2
    
    for r in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Create a random box filter initialization
        # Choose a random cutoff frequency
        # Lower cutoffs (smoother time domain) might be better for minimizing peaks?
        # Or higher cutoffs (noisier) might spread energy?
        # Let's sample cutoffs from a range.
        cutoff = np.random.randint(5, N // 2)
        F_init = np.zeros(N, dtype=np.complex128)
        
        # Fill the box with random phases
        F_init[0] = 1.0
        for k in range(1, cutoff):
            phase = np.random.uniform(0, 2 * np.pi)
            F_init[k] = np.exp(1j * phase)
            F_init[N - k] = np.conj(F_init[k])
        
        # Ensure conjugate symmetry
        for k in range(1, half + 1):
            F_init[N - k] = np.conj(F_init[k])
        
        # Coordinate descent on frequency domain
        F_opt, score_opt = coordinate_descent(F_init, start_time, TIME_LIMIT - 5, n_iter=500)
        
        # Convert back to time domain
        f_opt = np.real(np.fft.ifft(F_opt))
        f_opt = np.maximum(f_opt, 0.0)
        
        # Compute actual score
        actual_score = compute_score(f_opt)
        
        if actual_score < best_score:
            best_score = actual_score
            best_f = f_opt
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
