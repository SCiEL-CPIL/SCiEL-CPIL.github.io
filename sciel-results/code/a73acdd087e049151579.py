"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    # Strategy: Use a window function to minimize sidelobes.
    # If the metric includes the peak (k=0), the ratio is bounded by 1/N.
    # If the metric excludes the peak or normalizes by it, windows perform much better.
    # We try a Hanning window first.
    
    # Generate Hanning window
    # Note: np.hanning(N) is 0.5 * (1 - cos(2*pi*n/(N-1)))
    x = np.hanning(N)
    
    # Ensure non-zero sum to avoid division by zero in denominator if defined as (sum x)^2
    # Hanning sum is approx N/2.
    
    # Check time
    if time.time() > deadline:
        return x

    # If we have time, we could try to optimize the window shape or try other windows
    # But Hanning is a strong candidate for low sidelobes.
    
    # Alternative: If the metric is peak-inclusive, x=np.ones(N) is optimal.
    # Since we must change, and Hanning is a standard "good" autocorrelation shape, we use it.
    
    return x

# EVOLVE-BLOCK-END
