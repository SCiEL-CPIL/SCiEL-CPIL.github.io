import numpy as np
import time

N = 400
# Use a power of 2 for FFT efficiency, larger than 2*N-1
FFT_SIZE = 1024 

def _compute_peak(x):
    """
    Computes the maximum absolute value of the linear autocorrelation
    for lags k != 0.
    """
    # Zero pad to avoid circular correlation artifacts
    x_padded = np.zeros(FFT_SIZE)
    x_padded[:N] = x
    
    # Compute FFT
    X = np.fft.rfft(x_padded)
    
    # Power Spectrum
    P = X * np.conjugate(X)
    
    # Inverse FFT to get autocorrelation
    # Note: ifft of P gives circular autocorrelation of the padded signal.
    # Since we padded with zeros, the first N values correspond to positive lags of linear autocorrelation.
    R = np.fft.irfft(P, FFT_SIZE)
    
    # R[0] is the DC term (sum of squares = N)
    # R[1:N] are the positive lags
    # We want max |R[k]| for k in 1..N-1
    # Due to symmetry, R[-k] = R[k], and in the circular sense, the negative lags are at the end.
    # But for linear autocorrelation of a signal of length N padded to L:
    # R[k] for k=0..N-1 are the positive lags.
    # We only need to check 1..N-1.
    
    # Extract the relevant lags
    lags = R[1:N]
    
    return np.max(np.abs(lags))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    best_x = None
    best_score = float('inf')
    
    # Use a fixed seed for reproducibility in testing, but random in production?
    # The prompt implies finding a good sequence. Random restarts are standard.
    rng = np.random.default_rng(42)
    
    num_restarts = 50 # We will break early if time runs out
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with random +/- 1
        x = rng.choice([-1, 1], N)
        
        current_score = _compute_peak(x)
        
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
            
        # Local Search: Best Improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_flip_idx = -1
            best_new_score = current_score
            
            # Try flipping each bit and see if it improves the peak
            for i in range(N):
                old_val = x[i]
                x[i] = -old_val
                
                new_score = _compute_peak(x)
                
                # Revert immediately to keep x intact for next trials
                x[i] = old_val
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_flip_idx = i
            
            if best_flip_idx != -1:
                # Apply the best flip
                x[best_flip_idx] = -x[best_flip_idx]
                current_score = best_new_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
                    
        # Optional: Simulated Annealing step?
        # Given the time constraint, simple hill climbing with restarts is safer.
        # But we can try a few random flips that don't improve to escape local minima.
        # Let's add a simple escape mechanism: if stuck, try 10 random flips that might increase score slightly.
        # Actually, the restarts handle the global search.
        
    if best_x is None:
        # Fallback to seed if no time was spent
        return np.ones(N)
        
    return best_x

# EVOLVE-BLOCK-START
# The code above is the complete program.
# The construct function is the entry point.
# EVOLVE-BLOCK-END
