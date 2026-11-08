"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3(h):
    """Compute the autoconvolution upper bound C3 for a signed step function.
    
    C3 is defined as the maximum absolute value of the autoconvolution
    over all lags, normalized appropriately. For a step function with 
    equal intervals, this is proportional to max_m |sum_i h[i]*h[i+m]|.
    """
    # Compute autocorrelation for all lags
    # h is a 1D array of length N
    # Autoconvolution at lag m: sum_{i=0}^{N-1-m} h[i] * h[i+m]
    N = len(h)
    max_val = 0.0
    for m in range(N):
        if m == 0:
            # Lag 0: sum of squares
            val = np.dot(h, h)
        else:
            val = np.dot(h[:N-m], h[m:])
        abs_val = abs(val)
        if abs_val > max_val:
            max_val = abs_val
    return max_val

def _compute_c3_fast(h):
    """Faster computation using FFT-based autocorrelation."""
    N = len(h)
    # Zero-pad to avoid circular convolution
    size = 2 * N
    H = np.fft.fft(h, n=size)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return np.max(np.abs(ac))

def _objective(h):
    """Objective function to minimize."""
    return _compute_c3_fast(h)

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit, use 120 to be safe
    
    # Initialize with random signed steps
    np.random.seed(42)
    h = np.random.choice([-1.0, 1.0], size=N)
    
    best_h = h.copy()
    best_score = _objective(best_h)
    
    # Greedy / local search: try flipping individual elements
    # and small blocks to improve the objective
    
    def try_flip(h, idx):
        h[idx] *= -1.0
        new_score = _objective(h)
        h[idx] *= -1.0
        return new_score
    
    # Phase 1: Single-element flips
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if time.time() > deadline - 10:
                break
            h[i] *= -1.0
            new_score = _objective(h)
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
                improved = True
            else:
                h[i] *= -1.0
        
        if time.time() > deadline - 10:
            break
    
    # Phase 2: Block flips (flip consecutive blocks of size 2-10)
    for block_size in [2, 3, 4, 5, 6, 8, 10]:
        if time.time() > deadline - 10:
            break
        for i in range(0, N - block_size + 1):
            if time.time() > deadline - 10:
                break
            h[i:i+block_size] *= -1.0
            new_score = _objective(h)
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
            else:
                h[i:i+block_size] *= -1.0
    
    # Phase 3: Simulated annealing for more exploration
    if time.time() < deadline - 20:
        T = best_score * 0.01
        T_min = best_score * 0.0001
        cooling = 0.999
        
        current_h = best_h.copy()
        current_score = best_score
        
        while time.time() < deadline - 5:
            # Random move: flip a random block
            block_size = np.random.randint(1, 15)
            idx = np.random.randint(0, N - block_size + 1)
            
            current_h[idx:idx+block_size] *= -1.0
            new_score = _objective(current_h)
            
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = current_h.copy()
            else:
                current_h[idx:idx+block_size] *= -1.0
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    return best_h
# EVOLVE-BLOCK-END
