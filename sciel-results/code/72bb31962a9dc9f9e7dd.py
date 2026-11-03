"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Discrete spectral search with smooth initializations.
We parameterize the step function by its DFT coefficients.
We enforce non-negativity by clipping the time-domain signal.
We minimize the maximum value of the autoconvolution using coordinate descent
on the frequency domain coefficients.

Key improvements:
1. Smooth initializations (Cosine, Triangle, Gaussian) instead of random phases.
   Random phases lead to noisy time-domain signals, which when clipped become sparse
   and spiky, resulting in high autoconvolution peaks. Smooth signals remain dense
   after clipping and have lower peaks.
2. Improved coordinate descent:
   - Randomized frequency order to avoid cyclic dependencies.
   - "Best of two" update: Try +step and -step for each frequency, keep the best
     improvement rather than the first one.
   - Adaptive step size with decay.
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

def coordinate_descent(F_init, start_time, max_time, n_iter=1000):
    """
    Coordinate descent on DFT coefficients.
    Uses randomized order and 'best of two' updates.
    """
    current_F = F_init.copy()
    best_F = F_init.copy()
    best_score = objective_from_F(F_init)
    current_score = best_score
    
    half = N // 2
    step_size = 0.5
    no_improve_count = 0
    max_no_improve = 20
    
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        improved_in_pass = False
        # Randomize frequency order
        order = np.random.permutation(half + 1)
        
        for k in order:
            # Try +step and -step
            best_delta = 0
            best_trial_score = current_score
            
            # Trial +
            old_k = current_F[k]
            old_Nk = current_F[N - k] if k != 0 and k != half else None
            
            current_F[k] = old_k + step_size
            if k != 0 and k != half:
                current_F[N - k] = old_Nk + step_size
                
            score_plus = objective_from_F(current_F)
            if score_plus < best_trial_score:
                best_trial_score = score_plus
                best_delta = step_size
            
            # Trial -
            current_F[k] = old_k - step_size
            if k != 0 and k != half:
                current_F[N - k] = old_Nk - step_size
                
            score_minus = objective_from_F(current_F)
            if score_minus < best_trial_score:
                best_trial_score = score_minus
                best_delta = -step_size
            
            # Apply best delta
            if best_delta != 0:
                current_F[k] = old_k + best_delta
                if k != 0 and k != half:
                    current_F[N - k] = old_Nk + best_delta
                current_score = best_trial_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_F = current_F.copy()
                    improved_in_pass = True
                    no_improve_count = 0
            else:
                # Revert if no improvement
                current_F[k] = old_k
                if k != 0 and k != half:
                    current_F[N - k] = old_Nk
            
        if not improved_in_pass:
            no_improve_count += 1
            step_size *= 0.8
            if step_size < 1e-5 or no_improve_count > max_no_improve:
                break
        else:
            # Slightly increase step if we are making progress
            step_size = min(step_size * 1.05, 2.0)

    return best_F, best_score

def construct():
    start_time = time.time()
    
    # List of initializations
    inits = []
    
    # 1. Constant
    inits.append(np.ones(N))
    
    # 2. Cosine Bell (Hann Window)
    i = np.arange(N)
    hann = 0.5 * (1 + np.cos(2 * np.pi * i / N))
    inits.append(hann)
    
    # 3. Triangle
    tri = 1 - 2 * np.abs(i - N/2) / N
    tri[tri < 0] = 0
    inits.append(tri)
    
    # 4. Gaussian (several widths)
    for sigma in [N/4, N/6, N/8]:
        gauss = np.exp(-0.5 * ((i - N/2) / sigma) ** 2)
        inits.append(gauss)
        
    # 5. Two separated Cosine Bells
    bump1 = np.zeros(N)
    bump2 = np.zeros(N)
    idx1 = np.arange(max(0, N//4 - N//8), min(N, N//4 + N//8))
    if len(idx1) > 0:
        x1 = (idx1 - N/4) / (N/8)
        bump1[idx1] = 0.5 * (1 + np.cos(np.pi * x1))
    idx2 = np.arange(max(0, 3*N//4 - N//8), min(N, 3*N//4 + N//8))
    if len(idx2) > 0:
        x2 = (idx2 - 3*N/4) / (N/8)
        bump2[idx2] = 0.5 * (1 + np.cos(np.pi * x2))
    inits.append(bump1 + bump2)
    
    best_f = inits[0].copy()
    best_score = compute_score(best_f)
    
    # Convert inits to Frequency Domain for coordinate descent
    for f_init in inits:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        F_init = np.fft.fft(f_init)
        
        # Run coordinate descent
        F_opt, score_opt = coordinate_descent(F_init, start_time, TIME_LIMIT - 5, n_iter=200)
        
        # Convert back to time domain
        f_opt = np.real(np.fft.ifft(F_opt))
        f_opt = np.maximum(f_opt, 0.0)
        
        # Compute actual score (re-verification)
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
