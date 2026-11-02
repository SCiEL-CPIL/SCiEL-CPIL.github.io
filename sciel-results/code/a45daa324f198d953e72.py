"""Frequency-Domain Simulated Annealing for Autoconvolution Minimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 115  # safety margin before 120s kill
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _freq_to_time(log_mag):
    """
    Reconstruct time-domain signal from log-magnitude spectrum.
    log_mag: array of length N//2 + 1 (DC to Nyquist).
    We assume zero phase for simplicity, which yields a real, even (periodic) signal.
    To get a non-periodic-like signal, we can window it, but here we just use the IFFT.
    """
    mag = np.exp(log_mag)
    # Construct full complex spectrum
    # rfft returns N//2 + 1 values. We need to mirror them for ifft.
    # X[0] is real, X[N//2] is real (if N even).
    X = np.zeros(N, dtype=complex)
    X[:len(log_mag)] = mag
    # Mirror for negative frequencies
    if N % 2 == 0:
        # X[N//2] is Nyquist, stays real
        X[N//2 + 1:] = np.conj(mag[1:N//2][::-1])
    else:
        X[N//2 + 1:] = np.conj(mag[1:][::-1])
    
    x = np.fft.ifft(X).real
    return x

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x

    rng = np.random.default_rng(42)
    
    # Initialize with a Gaussian spectrum in frequency domain
    # This corresponds to a Sinc-like function in time, which has low sidelobes.
    k = np.arange(N // 2 + 1)
    kf = k / N
    # a controls bandwidth. Smaller a -> wider spectrum -> narrower time pulse -> higher peaks in conv?
    # Actually, we want to minimize max|ac|. 
    # Let's start with a moderate Gaussian.
    a_init = 1.5
    log_mag_current = -a_init * (kf**2)
    # Normalize log_mag so that sum(x) is not too small
    x_current = _freq_to_time(log_mag_current)
    if np.sum(x_current) < 1e-6:
        x_current += 1e-6
    
    current_val = _objective(x_current)
    if current_val < best_val:
        best_val = current_val
        best_x = x_current.copy()

    # Simulated Annealing Parameters
    T_start = 0.1
    T_end = 1e-6
    cooling_rate = 0.999
    T = T_start
    
    current_x = x_current.copy()
    current_log_mag = log_mag_current.copy()
    
    # Pre-compute FFT size for speed in _objective? _objective does it internally.
    # We can optimize the inner loop by reusing FFT if possible, but _objective is a black box for now.
    # To speed up, we can inline the objective calculation in the loop.
    
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # Fast objective calculation
    def _fast_objective(x):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10
        X = np.fft.rfft(x, n=L_fft)
        C = np.fft.irfft(X * X, n=L_fft)[:L]
        return np.max(np.abs(C)) / (s * s)
    
    # Update best with fast objective
    current_val = _fast_objective(current_x)
    if current_val < best_val:
        best_val = current_val
        best_x = current_x.copy()

    steps = 0
    max_steps = 200000 # Limit steps to ensure we finish
    
    while T > T_end and not _check_deadline():
        if steps >= max_steps:
            break
        steps += 1
        
        # Perturb log-magnitudes
        # Add Gaussian noise
        noise = rng.normal(0, 0.1, size=len(log_mag_current))
        trial_log_mag = current_log_mag + noise
        
        # Reconstruct x
        trial_x = _freq_to_time(trial_log_mag)
        
        # Evaluate
        trial_val = _fast_objective(trial_x)
        
        # Accept/Reject
        if trial_val < current_val:
            current_log_mag = trial_log_mag
            current_x = trial_x
            current_val = trial_val
            if current_val < best_val:
                best_val = current_val
                best_x = current_x.copy()
        else:
            # Metropolis criterion
            if trial_val - current_val < 0: # This is redundant with above, but for clarity
                accept = True
            else:
                # P(accept) = exp(-delta / T)
                # delta = trial_val - current_val
                # To avoid underflow, we can use log
                delta = trial_val - current_val
                if delta / T < -700: # exp(-700) is ~ 0
                    accept = False
                else:
                    accept = rng.random() < np.exp(-delta / T)
            
            if accept:
                current_log_mag = trial_log_mag
                current_x = trial_x
                current_val = trial_val
        
        # Cool down
        T *= cooling_rate
        
        # Adaptive noise level
        # If we haven't improved in a while, increase noise?
        # Simple approach: keep noise constant or decay it slightly.
        # Let's decay noise slightly to refine.
        # But we are using fixed 0.1. Let's make it adaptive.
        
    # Final verification with exact objective (just in case)
    final_val = _objective(best_x)
    if final_val < best_val:
        best_val = final_val

    return best_x
# EVOLVE-BLOCK-END
