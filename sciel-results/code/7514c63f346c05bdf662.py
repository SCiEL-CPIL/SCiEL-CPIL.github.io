"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Approach: Represent h as squared magnitude of analytic signal z.
    # We optimize the shape of h to flatten the autocorrelation g.
    # The inverse of a Hann window (high at ends, low in middle) 
    # tends to spread the autocorrelation energy more evenly 
    # compared to a bell shape, potentially reducing the peak-to-average ratio 
    # in the denominator (norm_inf) more effectively than the numerator (l2) penalty.
    
    t = np.linspace(0, 1, N)
    # Inverse Hann (raised cosine inverted): 0.5 * (1 + cos(2*pi*t))
    # This is 1 at the ends and 0 in the center.
    h = 0.5 * (1 + np.cos(2 * np.pi * t))
    
    # Ensure nonnegative and finite
    h = np.maximum(h, 0.0)
    
    # The magnitude scaling cancels out in the ratio, 
    # but we keep it normalized for stability.
    # Normalize to sum 1 or similar? 
    # Actually, let's just return the shape.
    return list(h)
# EVOLVE-BLOCK-END
