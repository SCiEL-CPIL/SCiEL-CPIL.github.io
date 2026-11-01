"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Start with alternating signs - this creates cancellation in autoconvolution
    # For a step function, autoconvolution peak is reduced by sign alternation
    heights = np.ones(N)
    heights[::2] = -1.0  # alternate: -1, 1, -1, 1, ...
    
    # Try a few different patterns and pick the best
    best_heights = heights.copy()
    best_score = _compute_c3_bound(heights)
    
    # Try block patterns: groups of same-sign values
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        for block_size in [1, 2, 3, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200, 400]:
            if time.time() > deadline - 5.0:
                break
            h = np.ones(N)
            for i in range(N):
                block_idx = i // block_size
                if block_idx % 2 == 0:
                    h[i] = -1.0
                else:
                    h[i] = 1.0
            s = _compute_c3_bound(h)
            if s < best_score:
                best_score = s
                best_heights = h.copy()
    
    # Try random perturbations of the best pattern
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        rng = np.random.RandomState(42)
        for trial in range(200):
            if time.time() > deadline - 5.0:
                break
            h = best_heights.copy()
            # Flip a random subset of signs
            mask = rng.rand(N) < 0.1
            h[mask] *= -1.0
            s = _compute_c3_bound(h)
            if s < best_score:
                best_score = s
                best_heights = h.copy()
    
    return best_heights


def _compute_c3_bound(heights):
    """Compute the upper bound on C3 for a signed step function.
    
    For a step function with N equal intervals, the autoconvolution is computed
    via discrete convolution. C3 is bounded by the L-infinity norm of the 
    autoconvolution (or a related moment).
    """
    # Discrete autoconvolution (scaled)
    # For step functions with equal width, the continuous autoconvolution
    # at grid points is proportional to the discrete convolution
    conv = np.convolve(heights, heights, mode='full')
    # The autoconvolution peak
    peak = np.max(np.abs(conv))
    # L1 norm squared (normalization)
    l1 = np.sum(np.abs(heights))
    # The upper bound on C3 is the peak autoconvolution
    # Normalized: we want to minimize peak / l1^2 or just peak
    # The "upper bound" is the actual autoconvolution maximum
    return peak
# EVOLVE-BLOCK-END
