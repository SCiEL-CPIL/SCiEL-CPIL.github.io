"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _fast_autocorr_max_abs(x):
    """
    Compute max absolute value of linear autocorrelation using FFT.
    x: array of signed values (+1/-1)
    """
    n = len(x)
    # For linear convolution/autocorrelation, we need length >= 2n - 1
    # FFT size should be power of 2 for speed, at least 2n
    size = 1
    while size < 2 * n:
        size *= 2
        
    # Pad x with zeros
    x_padded = np.zeros(size, dtype=np.float64)
    x_padded[:n] = x
    
    # FFT
    X = np.fft.fft(x_padded)
    # Autocorrelation is inverse FFT of |X|^2
    # Note: Linear autocorrelation c_k = sum x_i x_{i+k}
    # This is equivalent to circular autocorrelation if padded sufficiently.
    ac = np.fft.ifft(X * np.conj(X)).real
    
    # The first n values of ac correspond to lags 0 to n-1
    # Due to floating point errors, imaginary part is negligible
    # We only care about lags 0 to n-1 (positive lags)
    # Note: ac[0] is the sum of squares = n
    # The max absolute value usually occurs at small lags or specific structures
    # We take max of absolute values of the first n lags
    max_val = np.max(np.abs(ac[:n]))
    return max_val

def _walsh_hadamard_init(n):
    """
    Generate a sequence based on Walsh-Hadamard matrix properties.
    This provides a structured low-autocorrelation starting point.
    """
    # Find next power of 2 >= n
    size = 1
    while size < n:
        size *= 2
        
    # Generate Hadamard matrix H of size 'size'
    # H[0] = [1]
    # H[2k] = [H[k], H[k]]
    # H[2k+1] = [H[k], -H[k]]
    H = np.array([[1]], dtype=np.float64)
    while H.shape[0] < size:
        H = np.vstack([np.hstack([H, H]), np.hstack([H, -H])])
    
    # Take the first n rows or columns? 
    # Usually rows of Hadamard have zero autocorrelation for non-zero lags (for length n).
    # Let's take the first n elements of the first non-trivial row (index 1) 
    # or a random row to avoid trivial all-ones.
    # Row 0 is all ones. Row 1 is [1, -1, 1, -1...].
    # Let's pick a row that is not trivial.
    # For n=400, size=512.
    # We need a sequence of length 400.
    # We can take a random row from H, then truncate to n.
    # Or better, generate a random sign sequence but initialize with a 
    # "balanced" structure.
    
    # Actually, simple random initialization is often good enough if the search is strong.
    # But Walsh gives a better local structure.
    # Let's use a random row of the Hadamard matrix, truncated to n.
    idx = random.randint(1, size - 1)
    x = H[idx, :n].astype(np.int8)
    
    # Ensure it's +1/-1
    x[x == 0] = 1 # Just in case, though Hadamard is non-zero
    
    return x

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin
    
    # Initialize with a structured sequence
    x = _walsh_hadamard_init(N)
    best_val = _fast_autocorr_max_abs(x)
    best_x = x.copy()
    
    # Simulated Annealing / Stochastic Local Search
    # Temperature starts high, cools down
    T = 10.0
    cooling = 0.999
    
    # We will perform random single-bit flips
    # To speed up, we can batch flips or just rely on fast eval
    
    while time.time() < deadline:
        # Check time
        if time.time() > deadline - 2.0:
            break
            
        # Pick a random index to flip
        i = random.randint(0, N - 1)
        
        # Flip
        x[i] = -x[i]
        
        # Evaluate
        val = _fast_autocorr_max_abs(x)
        
        # Acceptance criterion
        if val < best_val:
            best_val = val
            best_x = x.copy()
            # Accept good move
            # Reset temperature slightly? Or keep cooling?
            # Usually keep cooling.
        else:
            # Probability of accepting worse move
            delta = val - best_val
            if delta > 0:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    # Accept
                    pass
                else:
                    # Reject: undo flip
                    x[i] = -x[i]
        
        # Cool down
        T *= cooling
        if T < 0.01:
            # Reheat occasionally to escape deep local minima?
            # Or just stop cooling and do local search?
            # Let's reheat slightly if we haven't improved for a while?
            # Simple strategy: if T is too low, just do greedy local search
            # But for now, just keep going with low T (greedy-ish)
            pass

    return best_x
# EVOLVE-BLOCK-END
