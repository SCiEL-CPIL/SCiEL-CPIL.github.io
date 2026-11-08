"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _compute_cost_fft(f):
    """
    Compute the cost J = max_k |AC(k)| / (sum(f))^2 using FFT.
    """
    s = np.sum(f)
    if s == 0:
        return float('inf')
    
    # Zero-pad to at least 2*N to avoid circular convolution artifacts
    # Linear autocorrelation length is 2*N - 1
    n_fft = 1 << (2 * N - 1).bit_length()
    
    F = np.fft.fft(f, n=n_fft)
    P = F * np.conj(F)
    ac = np.fft.ifft(P).real
    
    # The linear autocorrelation is in the first 2*N-1 elements
    # np.correlate(f, f, 'full') returns length 2*N-1
    # The FFT result corresponds to circular correlation of the padded array.
    # For zero-padded arrays, the first N elements and the last N-1 elements 
    # correspond to the linear autocorrelation.
    # Specifically, ac[0:N] are lags 0 to N-1.
    # ac[N_fft-N+1 : N_fft] are lags -(N-1) to -1.
    
    # Let's just take the max absolute value of the relevant part.
    # Actually, for zero-padded signal, the full linear autocorrelation is:
    # ac_lin = np.concatenate([ac[N_fft - (N-1):], ac[0:N]])
    
    ac_lin = np.concatenate([ac[N_fft - (N-1):], ac[0:N]])
    
    max_abs_ac = np.max(np.abs(ac_lin))
    
    return max_abs_ac / (s * s)

def construct():
    """
    Construct a signed step function using Simulated Annealing with FFT acceleration.
    """
    np.random.seed(42)
    
    # Start with random +/- 1
    f = np.random.choice([-1.0, 1.0], size=N)
    
    # Ensure sum is not zero
    while np.sum(f) == 0:
        idx = np.random.randint(N)
        f[idx] *= -1
        
    current_cost = _compute_cost_fft(f)
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.95
    min_T = 1e-4
    max_time = 110.0 # Leave 10s buffer
    start_time = time.time()
    
    # Precompute n_fft for speed
    n_fft = 1 << (2 * N - 1).bit_length()
    
    while T > min_T:
        if time.time() - start_time > max_time:
            break
            
        # Propose a move
        # Mix of single bit flips and block flips
        if np.random.random() < 0.8:
            # Single bit flip
            i = np.random.randint(N)
            f[i] *= -1.0
            move_type = 'bit'
        else:
            # Block flip
            length = np.random.randint(1, 11) # 1 to 10
            start = np.random.randint(N)
            indices = np.arange(start, start + length) % N
            f[indices] *= -1.0
            move_type = 'block'
            
        new_cost = _compute_cost_fft(f)
        
        # Acceptance criterion
        delta = new_cost - current_cost
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            current_cost = new_cost
        else:
            # Revert
            if move_type == 'bit':
                f[i] *= -1.0
            else:
                f[indices] *= -1.0
                
        T *= cooling_rate
        
    # Final local search (Steepest Descent) to polish
    start_time_ls = time.time()
    improved = True
    while improved and (time.time() - start_time_ls < 10.0):
        improved = False
        for i in range(N):
            f[i] *= -1.0
            new_cost = _compute_cost_fft(f)
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
            else:
                f[i] *= -1.0
                
    return f
# EVOLVE-BLOCK-END
