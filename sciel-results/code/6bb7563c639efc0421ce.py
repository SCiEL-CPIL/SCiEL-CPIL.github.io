import numpy as np
import time

N = 400

def _compute_psl(x):
    """
    Compute the Peak Sidelobe Level (PSL) of the autocorrelation of x.
    x: numpy array of length N with values +/- 1.
    Returns: max_{k != 0} |R(k)|
    """
    # Compute autocorrelation via FFT
    # R[k] = sum_{i} x[i] x[i+k]
    # Using FFT: ifft(|fft(x)|^2) gives circular autocorrelation.
    # For linear autocorrelation of length N, we need to be careful with wrap-around.
    # However, for PSL of a finite sequence, circular autocorrelation is a common approximation 
    # or we can zero-pad. 
    # Zero-padding to 2N avoids circular aliasing for lags up to N-1.
    x_padded = np.zeros(2 * N, dtype=float)
    x_padded[:N] = x
    fft_x = np.fft.fft(x_padded)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real
    # The ACF is centered. For zero-padded input of size 2N, 
    # the peak (lag 0) is at index N.
    # Lags 1 to N-1 are at indices N+1 to N+N-1 (which wraps? No, ifft returns 0..2N-1).
    # Actually, ifft of |fft|^2 gives circular correlation.
    # With zero padding to 2N, the linear correlation is embedded in the first N elements?
    # Let's check indices.
    # x = [x0, x1, ..., x_{N-1}, 0, ..., 0]
    # R[k] for k>0 is sum x[i]x[i+k].
    # The standard trick:
    # r = ifft(fft(x) * conj(fft(x)))
    # This gives circular correlation.
    # If we zero pad to 2N, the linear correlation lags -N+1 to N-1 are in the array.
    # Specifically, r[N] is lag 0.
    # r[N+1] is lag 1.
    # r[N-1] is lag -1.
    # We want max |r[k]| for k in 1..N-1.
    
    # r has length 2N.
    # Lag 0 is at index N.
    # Lag 1 is at index N+1.
    # Lag N-1 is at index 2N-1.
    # Lag -1 is at index N-1.
    # Lag -(N-1) is at index 1.
    
    # So sidelobes are indices 1..N-1 and N+1..2N-1.
    # Due to symmetry for real x, r[k] = r[-k].
    # So we just need to check indices N+1 to 2N-1 (lags 1 to N-1).
    
    sidelobes = acf[N+1:] # lags 1 to N-1
    return np.max(np.abs(sidelobes))

def construct():
    start_time = time.time()
    deadline = 100.0  # Leave 20s buffer for safety
    
    best_x = None
    best_score = float('inf')
    
    # Initialize with random sequence
    # Or try a few random starts and pick the best initial one
    x = np.random.choice([-1, 1], size=N)
    current_score = _compute_psl(x)
    best_x = x.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # We will try single bit flips and small block flips
    # Precompute indices? No, just randomize.
    
    while time.time() < deadline:
        if T < min_T:
            T = 1.0 # Reheat or restart
            # Optionally restart from best_x
            # x = best_x.copy()
            # current_score = best_score
            
        # Generate a move
        # 80% single bit, 20% block flip (size 2-5)
        if np.random.random() < 0.8:
            idx = np.random.randint(N)
            x_new = x.copy()
            x_new[idx] *= -1
        else:
            block_size = np.random.randint(2, 6)
            start_idx = np.random.randint(N - block_size + 1)
            x_new = x.copy()
            x_new[start_idx:start_idx+block_size] *= -1
        
        new_score = _compute_psl(x_new)
        
        # Acceptance criterion
        if new_score < current_score:
            x = x_new
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
        else:
            # Accept with probability exp(-delta / T)
            delta = new_score - current_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    x = x_new
                    current_score = new_score
            else:
                # This case is covered by new_score < current_score, 
                # but if they are equal, accept to move around
                x = x_new
                current_score = new_score
        
        T *= cooling_rate
        
    return best_x
