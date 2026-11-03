"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Discrete spectral search using L-BFGS-B optimization on DFT coefficients.
We parameterize the signal by its DFT coefficients, ensuring conjugate symmetry.
We use a soft-plus projection to handle non-negativity smoothly during optimization.
The objective function is the normalized maximum of the autoconvolution.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def softplus(x, beta=10.0):
    """Numerically stable softplus."""
    return np.logaddexp(0, beta * x) / beta

def dft_coeffs_to_signal(F_flat):
    """
    Convert a flat array of real/imag parts of the first half of DFT coefficients
    to a time-domain signal.
    F_flat structure: [Re(F[0]), Im(F[1]), Re(F[1]), ..., Re(F[N/2])]
    Note: F[0] is real, F[N/2] is real for even N.
    """
    half = N // 2
    # Reconstruct F
    F = np.zeros(N, dtype=np.complex128)
    # F[0]
    F[0] = F_flat[0]
    # F[1] to F[half-1]
    # F_flat[1] is Im(F[1]), F_flat[2] is Re(F[1])? 
    # Let's use a simpler mapping:
    # Indices 0..half-1 correspond to k=1..half-1
    # We store Re and Im for each k.
    # Let's just use a vector of length 2*(half-1) + 2 (for k=0 and k=half)
    
    # Let's redefine the parameter vector for clarity in the optimizer call.
    # We will pass a vector x of length 2*(N//2) 
    # x[0] = Re(F[0])
    # x[1] = Re(F[1]), x[2] = Im(F[1])
    # ...
    # x[2*(k)] = Re(F[k]), x[2*(k)+1] = Im(F[k]) for k=1..N//2-1
    # x[2*(N//2)] = Re(F[N//2])
    # Im(F[N//2]) is 0.
    
    # Total length: 1 + 2*(N//2 - 1) + 1 = N
    # Actually, let's just use N parameters: Re and Im for all N, but enforce symmetry in the objective?
    # No, better to optimize only independent parameters.
    
    # Independent params:
    # k=0: 1 param (Real)
    # k=1..N/2-1: 2 params each (Real, Imag)
    # k=N/2: 1 param (Real)
    # Total: 1 + 2*(N/2 - 1) + 1 = N params.
    
    F[0] = F_flat[0]
    for k in range(1, half):
        F[k] = F_flat[2*k - 1] + 1j * F_flat[2*k]
        F[N - k] = np.conj(F[k])
    F[half] = F_flat[2*half - 1] # Last real part
    # Note: F_flat indices:
    # 0: Re(F[0])
    # 1: Re(F[1]), 2: Im(F[1])
    # 3: Re(F[2]), 4: Im(F[2])
    # ...
    # 2*k-1: Re(F[k]), 2*k: Im(F[k])
    # For k=half: index 2*half - 1.
    
    f = np.real(np.fft.ifft(F))
    return f

def objective_wrapper(x):
    """
    Objective function for the optimizer.
    x: flat array of DFT parameters.
    """
    f = dft_coeffs_to_signal(x)
    # Soft projection to non-negative
    f_pos = softplus(f, beta=10.0)
    
    s = np.sum(f_pos)
    if s < 1e-15:
        return 1e10
        
    # Compute linear autoconvolution
    # For speed, we can use FFT for circular, but linear is required for exact score.
    # However, for large N, circular is a good approximation for the *middle* part,
    # but the peak is usually at lag 0, which is the same for linear and circular
    # if the signal doesn't wrap around significantly.
    # To be safe and fast, let's use np.convolve. N=600 is small enough.
    conv = np.convolve(f_pos, f_pos, mode='full')
    max_conv = np.max(conv)
    
    return max_conv / (s * s)

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Prepare initial DFT parameters for constant signal
    F_const = np.fft.fft(best_f)
    x0 = np.zeros(N)
    half = N // 2
    x0[0] = F_const[0].real
    for k in range(1, half):
        x0[2*k - 1] = F_const[k].real
        x0[2*k] = F_const[k].imag
    x0[2*half - 1] = F_const[half].real
    
    # Also try a Gaussian initialization
    t = np.arange(N)
    f_gauss = np.exp(-((t - N/2)**2) / (2 * (N/4)**2))
    F_gauss = np.fft.fft(f_gauss)
    x1 = np.zeros(N)
    x1[0] = F_gauss[0].real
    for k in range(1, half):
        x1[2*k - 1] = F_gauss[k].real
        x1[2*k] = F_gauss[k].imag
    x1[2*half - 1] = F_gauss[half].real
    
    # Random initialization
    np.random.seed(42)
    f_rand = np.random.rand(N)
    F_rand = np.fft.fft(f_rand)
    x2 = np.zeros(N)
    x2[0] = F_rand[0].real
    for k in range(1, half):
        x2[2*k - 1] = F_rand[k].real
        x2[2*k] = F_rand[k].imag
    x2[2*half - 1] = F_rand[half].real
    
    initializations = [x0, x1, x2]
    
    for x_init in initializations:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        try:
            # Use L-BFGS-B
            res = minimize(objective_wrapper, x_init, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.success:
                f_opt = dft_coeffs_to_signal(res.x)
                f_opt = np.maximum(f_opt, 0.0) # Hard project for final score
                score_opt = compute_score(f_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_f = f_opt
        except Exception:
            pass
            
    # Final local refinement in time domain if time permits
    if time.time() - start_time < TIME_LIMIT - 5:
        h = best_f.copy()
        alpha = 0.05
        for _ in range(100):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            s = np.sum(h)
            if s < 1e-12: break
            h_norm = h * (N / s)
            conv = np.convolve(h_norm, h_norm, mode='full')
            max_c = np.max(conv)
            current_val = max_c / (N * N)
            
            # Simple sub-gradient step: reduce values that contribute to peak
            # Peak is at lag 0 usually.
            # Reduce all values slightly, but less for those that are "isolated"?
            # Or just scale down the peak contributors.
            # Let's try a simple multiplicative update
            # h_new = h * (1 - alpha * (h / max(h)))
            # This flattens the distribution.
            h_max = np.max(h)
            if h_max > 0:
                h_new = h * (1.0 - alpha * (h / h_max))
                h_new = np.maximum(h_new, 0.0)
                s_new = np.sum(h_new)
                if s_new > 1e-12:
                    h_new = h_new * (N / s_new)
                    conv_new = np.convolve(h_new, h_new, mode='full')
                    val_new = np.max(conv_new) / (N * N)
                    if val_new < current_val:
                        h = h_new
                    else:
                        alpha *= 0.9
                else:
                    break
            else:
                break
                
        score_h = compute_score(h)
        if score_h < best_score:
            best_score = score_h
            best_f = h
            
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
