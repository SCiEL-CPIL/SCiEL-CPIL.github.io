"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Discrete spectral search.
Parameterize the step function by its DFT coefficients.
Enforce non-negativity via hard projection.
Directly minimize the maximum value of the autoconvolution using coordinate descent
on the frequency domain, with FFT-based autoconvolution computation for speed.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score_fast(f):
    """Compute the normalized maximum of the autoconvolution using FFT."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Autoconvolution via FFT
    F = np.fft.fft(f, n=2*N)
    conv_full = np.fft.ifft(F * F).real
    # The full convolution has length 2*N-1, but FFT gives 2*N
    # Take the first 2*N-1 elements
    conv = conv_full[:2*N-1]
    return np.max(conv) / (s * s)

def objective_from_F(F):
    """Compute the exact score from DFT coefficients using FFT-based autoconvolution."""
    f = np.real(np.fft.ifft(F))
    f = np.maximum(f, 0.0)
    return compute_score_fast(f)

def coordinate_descent_optimized(F, start_time, max_time, n_iter=2000):
    """Coordinate descent on DFT coefficients with FFT-based objective."""
    best_F = F.copy()
    best_score = objective_from_F(F)
    current_F = F.copy()
    
    half = N // 2
    step_size = 0.5
    no_improve_count = 0
    
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        improved_in_pass = False
        # Iterate over frequency bins in random order to avoid cycles
        order = np.random.permutation(half + 1)
        
        for k in order:
            if time.time() - start_time > max_time:
                break
            
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
                    
                    if score_trial < best_score - 1e-12:
                        best_score = score_trial
                        best_F = current_F.copy()
                        improved_in_pass = True
                        no_improve_count = 0
                        # Keep the change, increase step slightly
                        step_size = min(step_size * 1.1, 10.0)
                        break  # Move to next k
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
            if step_size < 1e-6 or no_improve_count > 50:
                break
    
    return best_F, best_score

def generate_initializations(n_restarts, start_time):
    """Generate diverse initializations for the frequency domain search."""
    inits = []
    half = N // 2
    
    # 1. Constant function (baseline)
    F_const = np.zeros(N, dtype=np.complex128)
    F_const[0] = N  # DC component for constant 1
    inits.append(F_const)
    
    # 2. Triangular pulse in time -> sinc-like in frequency
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    tri = np.maximum(0, 1 - np.abs(t) / 0.25)
    F_tri = np.fft.fft(tri)
    inits.append(F_tri)
    
    # 3. Gaussian-like in time
    sigma = 0.08
    gauss = np.exp(-t**2 / (2 * sigma**2))
    F_gauss = np.fft.fft(gauss)
    inits.append(F_gauss)
    
    # 4-10. Random box filters with different cutoffs
    for r in range(7):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        cutoff = np.random.randint(20, N // 2)
        F_init = np.zeros(N, dtype=np.complex128)
        F_init[0] = 1.0
        for k in range(1, cutoff):
            phase = np.random.uniform(0, 2 * np.pi)
            F_init[k] = np.exp(1j * phase)
            F_init[N - k] = np.conj(F_init[k])
        # Ensure conjugate symmetry for Nyquist
        if half < N:
            F_init[N - half] = np.conj(F_init[half])
        inits.append(F_init)
    
    # 11-15. Smoothed random spectra
    for r in range(5):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        # Random smooth spectrum
        mag = np.random.rand(N) ** 0.5
        # Make it conjugate symmetric
        F_init = np.zeros(N, dtype=np.complex128)
        F_init[0] = mag[0]
        for k in range(1, half + 1):
            phase = np.random.uniform(0, 2 * np.pi)
            F_init[k] = mag[k] * np.exp(1j * phase)
            if k < N - k:
                F_init[N - k] = mag[k] * np.exp(-1j * phase)
        inits.append(F_init)
    
    return inits

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score_fast(best_f)
    
    # Generate diverse initializations
    inits = generate_initializations(15, start_time)
    
    # Coordinate descent from each initialization
    for i, F_init in enumerate(inits):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Coordinate descent on frequency domain
        F_opt, score_opt = coordinate_descent_optimized(
            F_init, start_time, TIME_LIMIT - 3, n_iter=1500
        )
        
        # Convert back to time domain
        f_opt = np.real(np.fft.ifft(F_opt))
        f_opt = np.maximum(f_opt, 0.0)
        
        # Compute actual score
        actual_score = compute_score_fast(f_opt)
        
        if actual_score < best_score:
            best_score = actual_score
            best_f = f_opt
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score_fast(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
