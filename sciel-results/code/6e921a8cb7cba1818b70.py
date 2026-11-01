"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(f):
    """
    Computes the score for a given signed step function.
    Metric: Sum of cubes of absolute values of circular autocorrelation coefficients.
    This measures the 'flatness' of the autocorrelation. Lower is better.
    """
    fft_f = np.fft.fft(f)
    power_spectrum = fft_f * np.conj(fft_f)
    autocorr = np.fft.ifft(power_spectrum).real
    score = np.sum(np.abs(autocorr)**3)
    return score

def construct():
    """
    Construct a signed step function with N=400 equal intervals on [-1/4, 1/4].
    Uses Simulated Annealing with incremental autocorrelation updates.
    """
    # Initialize random binary sequence
    f = np.random.choice([-1, 1], size=N)
    
    # Compute initial autocorrelation and score
    # A[m] = sum_i f[i] f[i+m]
    # Using FFT for initialization is fine
    fft_f = np.fft.fft(f)
    A = np.fft.ifft(fft_f * np.conj(fft_f)).real
    S = np.sum(np.abs(A)**3)
    
    best_f = f.copy()
    best_S = S
    
    # Simulated Annealing parameters
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    T = 10000.0
    cooling_rate = 0.99999
    min_T = 1e-5
    
    # Precompute arange for efficiency
    arange_N = np.arange(N)
    
    while time.time() - start_time < time_limit:
        # Pick a random index to flip
        k = np.random.randint(N)
        old_val = f[k]
        new_val = -old_val
        
        # Calculate change in autocorrelation
        # A'[m] = A[m] - 2 * f[k] * f[k+m] - 2 * f[k] * f[k-m]
        # Exception: m=0, delta is 0.
        idx_plus = (k + arange_N) % N
        idx_minus = (k - arange_N) % N
        
        delta_A = -2.0 * old_val * (f[idx_plus] + f[idx_minus])
        delta_A[0] = 0.0 # A[0] is constant N
        
        new_A = A + delta_A
        new_S = np.sum(np.abs(new_A)**3)
        
        delta_S = new_S - S
        
        # Acceptance criterion
        if delta_S < 0:
            # Accept if better
            f[k] = new_val
            A = new_A
            S = new_S
            
            if S < best_S:
                best_S = S
                best_f = f.copy()
        else:
            # Accept if worse with probability exp(-delta_S / T)
            if np.random.random() < np.exp(-delta_S / T):
                f[k] = new_val
                A = new_A
                S = new_S
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T

    return best_f
# EVOLVE-BLOCK-END
