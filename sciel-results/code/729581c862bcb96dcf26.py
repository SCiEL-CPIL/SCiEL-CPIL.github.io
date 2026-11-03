"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a signed step function (vector of +1/-1) that minimizes
    the maximum absolute autoconvolution.
    
    We use a combination of:
    1. Random initialization with local search (simulated annealing / greedy)
    2. Fast vectorized evaluation using FFT for autoconvolution
    3. Deadline guard to ensure we finish within time limits
    """
    start_time = time.time()
    DEADLINE = 130  # seconds, leave margin for evaluator overhead
    
    # Use FFT to compute autoconvolution efficiently
    # For a vector x of length N, autoconvolution c[k] = sum_i x[i]*x[i+k]
    # This is equivalent to the circular convolution if we zero-pad, or linear conv
    
    def compute_max_abs_autocorr(x):
        """Compute max |autocorr| for a real vector x using FFT."""
        # Linear autocorrelation via FFT
        n = len(x)
        size = 2 * n
        # Use next power of 2 for efficiency
        import math
        size = 1
        while size < 2 * n:
            size *= 2
        
        X = np.fft.rfft(x, size)
        # Autocorrelation is inverse FFT of |X|^2
        ac = np.fft.irfft(X * np.conj(X), size)[:n]
        # ac[k] = sum_i x[i] * x[i+k] for k=0..n-1
        # But we need sum over valid indices only (linear autocorrelation)
        # Actually, the standard definition: R[k] = sum_{i=0}^{n-1-k} x[i]*x[i+k]
        # The FFT approach gives circular autocorrelation. For linear, we need to be careful.
        # Let's use a different approach: direct computation is O(N^2) which for N=400 is 160k ops - fast enough
        # Actually let's just do it directly since N=400 is small
        # R[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]
        # Max over k of |R[k]|
        
        # Direct O(N^2) computation
        # For N=400, this is 400*400/2 = 80k multiplications per evaluation
        # With vectorization this is very fast
        
        # Vectorized: for each lag k, compute dot product of x[:N-k] and x[k:]
        # We can do this with a loop over k, but let's vectorize the whole thing
        # Actually, let's precompute using the FFT method properly
        
        # Proper linear autocorrelation via FFT:
        # Zero-pad x to length >= 2N-1
        # Take FFT, multiply by conjugate, inverse FFT
        # The first N elements give the linear autocorrelation
        
        # Let me just use scipy if available, or do it directly
        # For simplicity and correctness, direct computation with numpy
        max_val = 0
        for k in range(n):
            # R[k] = sum_{i=0}^{n-1-k} x[i] * x[i+k]
            if k == 0:
                val = np.dot(x, x)
            else:
                val = np.dot(x[:n-k], x[k:])
            abs_val = abs(val)
            if abs_val > max_val:
                max_val = abs_val
        return max_val
    
    # Faster: use the FFT-based linear autocorrelation
    def compute_max_abs_autocorr_fast(x):
        """Fast linear autocorrelation max abs value using FFT."""
        n = len(x)
        # Next power of 2 >= 2n-1
        size = 1
        while size < 2 * n - 1:
            size *= 2
        
        X = np.fft.rfft(x, size)
        power = X * np.conj(X)
        ac = np.fft.irfft(power, size)
        # ac[0:n] contains the linear autocorrelation R[0], R[1], ..., R[n-1]
        # where R[k] = sum_{i=0}^{n-1-k} x[i]*x[i+k]
        # Wait, actually for real signals, the autocorrelation from FFT is:
        # ac[k] = sum_{i=0}^{size-1} x_padded[i] * x_padded[(i+k) % size]
        # For zero-padded x of length >= 2n-1, the first n values are the linear autocorrelation
        return np.max(np.abs(ac[:n]))
    
    # Test the fast method
    def test_autocorr():
        x = np.ones(10)
        # R[0] = 10, R[1] = 9, ..., R[9] = 1
        # Max should be 10
        val = compute_max_abs_autocorr_fast(x)
        assert abs(val - 10) < 1e-6, f"Expected 10, got {val}"
    
    # test_autocorr()
    
    # Initialize with random signs
    x = np.random.choice([-1, 1], size=N)
    best_x = x.copy()
    best_score = compute_max_abs_autocorr_fast(x)
    
    # Simulated annealing with vectorized evaluation
    # We'll flip individual bits or small blocks
    
    # Precompute: for flipping bit i, how does the autocorrelation change?
    # This is complex, so let's just use full re-evaluation but make it fast
    
    # Use a greedy local search first, then simulated annealing
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        for i in range(N):
            if time.time() > DEADLINE - 5:
                break
            # Try flipping bit i
            x[i] = -x[i]
            score = compute_max_abs_autocorr_fast(x)
            if score < best_score:
                best_score = score
                best_x = x.copy()
                improved = True
            else:
                x[i] = -x[i]  # revert
        # If no improvement in a full pass, break
    
    # Now do simulated annealing for more exploration
    if time.time() < DEADLINE - 10:
        T = 1.0
        cooling = 0.999
        x = best_x.copy()
        current_score = best_score
        
        while T > 0.01 and time.time() < DEADLINE - 5:
            # Pick a random position to flip
            i = np.random.randint(N)
            x[i] = -x[i]
            new_score = compute_max_abs_autocorr_fast(x)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_x = x.copy()
            else:
                # Accept with probability exp(-(new_score - current_score) / T)
                delta = new_score - current_score
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if np.random.random() < prob:
                        current_score = new_score
                    else:
                        x[i] = -x[i]  # revert
                else:
                    current_score = new_score
            
            T *= cooling
    
    # Final greedy polish
    if time.time() < DEADLINE - 2:
        x = best_x.copy()
        improved = True
        while improved and time.time() < DEADLINE - 1:
            improved = False
            for i in range(N):
                if time.time() > DEADLINE - 1:
                    break
                x[i] = -x[i]
                score = compute_max_abs_autocorr_fast(x)
                if score < best_score:
                    best_score = score
                    best_x = x.copy()
                    improved = True
                else:
                    x[i] = -x[i]  # revert
    
    return best_x
# EVOLVE-BLOCK-END
