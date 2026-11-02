"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _fast_autocorr_max_abs(x):
    """Compute max absolute value of linear autocorrelation using FFT."""
    n = len(x)
    # Pad to next power of 2 for efficiency
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    # Zero-pad x
    x_padded = np.zeros(size, dtype=np.float64)
    x_padded[:n] = x
    
    # FFT
    X = np.fft.fft(x_padded, n=size)
    
    # Autocorrelation is IFFT of |X|^2
    # Note: Linear autocorrelation is centered. 
    # The raw IFFT gives circular autocorrelation.
    # For linear autocorrelation, we need to extract the correct part.
    # However, for max absolute value, we can just compute the full circular autocorrelation 
    # and take the max absolute value. 
    # Wait, circular autocorrelation wraps around. 
    # Linear autocorrelation r[k] = sum x[i] x[i+k].
    # Circular autocorrelation c[k] = sum x[i] x[(i+k) mod N].
    # If we zero-pad to length >= 2N-1, the circular autocorrelation of the padded signal
    # equals the linear autocorrelation of the original signal for k in [-(N-1), N-1].
    # The FFT approach computes the circular autocorrelation of the padded signal.
    # Let's verify:
    # x_padded = [x0, x1, ..., xN-1, 0, ..., 0]
    # C = IFFT(|FFT(x_padded)|^2)
    # C[k] for k=0..N-1 is the linear autocorrelation r[k].
    # C[k] for k=N..2N-2 corresponds to negative lags.
    # So we can just take the max absolute value of C.
    
    C = np.fft.ifft(X * np.conj(X)).real
    
    # The autocorrelation is symmetric.
    # We just need the max absolute value of the entire array C.
    # Note: C[0] is the sum of squares, which is N (if x is +/-1).
    # The max absolute value will be at least N.
    # We want to minimize this.
    
    return np.max(np.abs(C))

def _objective(y):
    """Objective function for continuous optimization.
    y is a vector of length N with values in [-1, 1] (or larger, but we'll clip).
    """
    # Clip to [-1, 1] to keep it bounded? Or let it be free?
    # The problem is about signed step functions (heights +/-1).
    # Relaxing to continuous [-1, 1] is a standard approach.
    x = np.clip(y, -1, 1)
    return _fast_autocorr_max_abs(x)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin
    
    # Start with a random continuous vector
    np.random.seed(42)
    y0 = np.random.uniform(-1, 1, N)
    
    best_val = float('inf')
    best_x = None
    
    # Try a few random restarts
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
            
        y0 = np.random.uniform(-1, 1, N)
        
        # Use BFGS to minimize the max absolute autocorrelation
        # Note: _fast_autocorr_max_abs is non-smooth (due to max and abs), 
        # but BFGS can still make progress. 
        # A smoother objective might be sum of squares of autocorrelation, 
        # but the task asks for max absolute value.
        # Let's try minimizing the objective directly.
        
        try:
            res = minimize(_objective, y0, method='BFGS', 
                           options={'maxiter': 50, 'disp': False})
            y_opt = res.x
            x_cont = np.clip(y_opt, -1, 1)
            
            # Threshold to binary
            x_bin = np.sign(x_cont)
            # Ensure no zeros
            x_bin[x_bin == 0] = 1
            
            val = _fast_autocorr_max_abs(x_bin)
            
            if val < best_val:
                best_val = val
                best_x = x_bin.copy()
                
        except Exception as e:
            pass
            
        # Also try a simple local search on the best binary solution found so far
        if best_x is not None and time.time() < deadline - 10.0:
            x = best_x.copy()
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                # Randomly try flipping a few bits
                indices = np.random.choice(N, size=min(50, N), replace=False)
                for i in indices:
                    if time.time() > deadline - 5.0:
                        break
                    old = x[i]
                    x[i] = -old
                    val = _fast_autocorr_max_abs(x)
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        improved = True
                    else:
                        x[i] = old
                        x = best_x.copy() # Reset to best

    # If no binary solution was found, return all ones
    if best_x is None:
        return np.ones(N, dtype=np.int8)
        
    return best_x.astype(np.int8)
# EVOLVE-BLOCK-END
