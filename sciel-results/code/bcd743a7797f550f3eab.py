"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    def max_autocorr(h):
        """
        Compute the maximum absolute autocorrelation at non-zero lags.
        For a sequence of length N, the autocorrelation at lag k is sum_{i=0}^{N-k-1} h[i]*h[i+k].
        We want to minimize the maximum of |autocorr[k]| for k=1..N-1.
        """
        # Use FFT to compute autocorrelation efficiently
        # Zero-pad to at least 2*N-1 to avoid circular convolution effects
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        # acf[0] is the energy (sum of squares), which is N for ±1 sequences
        # acf[1:N] contains the autocorrelations for lags 1 to N-1
        # Due to numerical precision, we take the max absolute value
        return np.max(np.abs(acf[1:N]))
    
    # Start with a random ±1 sequence
    rng = np.random.RandomState(42)
    best_h = rng.choice([1, -1], N).astype(np.float64)
    best_val = max_autocorr(best_h)
    
    # Local search: flip bits to reduce the max autocorrelation
    def local_search(h, max_iters=50):
        """Perform local search by flipping bits."""
        current_val = max_autocorr(h)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try flipping each bit
            for i in range(N):
                if time.time() > deadline - 10.0:
                    return h, current_val
                h[i] *= -1
                new_val = max_autocorr(h)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                else:
                    h[i] *= -1  # revert
        return h, current_val
    
    # Multiple restarts with local search
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Generate a new random sequence
        current_h = rng.choice([1, -1], N).astype(np.float64)
        
        # Perform local search
        current_h, current_val = local_search(current_h, max_iters=30)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
    
    # If time remains, try a few more local search passes on the best solution
    if time.time() < deadline - 15.0:
        best_h, best_val = local_search(best_h.copy(), max_iters=100)
    
    return best_h
# EVOLVE-BLOCK-END
