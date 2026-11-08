import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Initialize with random signs
    # Using a fixed seed for reproducibility in debugging, but random is fine
    rng = np.random.default_rng(42)
    h = rng.choice([-1.0, 1.0], size=N)
    
    def compute_metric(h):
        # Autocorrelation via FFT
        # Zero pad to 2*N to avoid circular convolution artifacts
        fft_size = 1 << (2 * N - 1).bit_length()
        h_fft = np.fft.fft(h, n=fft_size)
        power_spectrum = h_fft * np.conjugate(h_fft)
        autocorr = np.fft.ifft(power_spectrum).real
        # Take the first N lags (0 to N-1) and their symmetric counterparts
        # R_k for k=0 to N-1
        # The full autocorrelation is symmetric.
        # We want sum |R_k| for all k.
        # Due to symmetry, sum_{k=-(N-1)}^{N-1} |R_k| = |R_0| + 2 * sum_{k=1}^{N-1} |R_k|
        return np.abs(autocorr[0]) + 2.0 * np.sum(np.abs(autocorr[1:N]))

    best_score = compute_metric(h)
    best_h = h.copy()
    
    # Greedy local search
    # Iterate until no improvement or time runs out
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            # Flip sign
            h[i] *= -1.0
            new_score = compute_metric(h)
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
                improved = True
            else:
                # Revert
                h[i] *= -1.0
                
    return best_h
# EVOLVE-BLOCK-END
