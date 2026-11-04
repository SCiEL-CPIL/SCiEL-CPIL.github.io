"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _rule90_sequence(seed_pos, k_steps, total_len):
    """Generate Rule 90 CA sequence from a single seed."""
    # Initialize with single 1 at seed_pos
    row = np.zeros(total_len, dtype=np.uint8)
    row[seed_pos] = 1
    
    for _ in range(k_steps):
        new_row = np.zeros(total_len, dtype=np.uint8)
        # Rule 90: new[i] = left[i] XOR right[i]
        # For boundaries, treat as 0
        left = np.roll(row, 1)
        right = np.roll(row, -1)
        new_row = (left ^ right) & 1
        row = new_row
    
    # Map 0 -> -1, 1 -> +1
    return 2 * row.astype(np.float64) - 1


def _autoconvolution_peak(x):
    """Compute the peak absolute value of the autoconvolution."""
    n = len(x)
    # Circular autoconvolution via FFT
    X = np.fft.fft(x, n=2*n)
    conv = np.fft.ifft(X * X).real
    # The autoconvolution has length 2n-1, but we take the circular version
    # Actually, let's compute linear autoconvolution
    conv = np.correlate(x, x, mode='full')
    return np.max(np.abs(conv))


def _local_search(x, max_iters=5000, time_limit=30):
    """Local search: flip individual bits to reduce autoconvolution peak."""
    x = x.copy()
    best_peak = _autoconvolution_peak(x)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        n = len(x)
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            # Try flipping position i
            x[i] *= -1
            new_peak = _autoconvolution_peak(x)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                improved = True
            else:
                x[i] *= -1  # revert
    return x, best_peak


def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Generate Rule 90 sequence
    # Use multiple seeds to fill the 400 positions
    # Try different seed positions and step counts
    best_x = None
    best_peak = np.inf
    
    # Try several configurations
    configs = [
        (200, 200),  # seed at 200, 200 steps
        (199, 199),
        (200, 199),
        (100, 100),
        (300, 100),
        (0, 200),
        (399, 200),
    ]
    
    for seed_pos, k_steps in configs:
        if (time.time() - start_time) > deadline - 30:
            break
        
        x = _rule90_sequence(seed_pos, k_steps, N)
        
        # Local search refinement
        x_refined, peak = _local_search(x, max_iters=3000, time_limit=15)
        
        if peak < best_peak:
            best_peak = peak
            best_x = x_refined
    
    if best_x is None:
        # Fallback
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
