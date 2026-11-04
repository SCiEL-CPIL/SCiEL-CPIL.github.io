"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Theoretical analysis suggests that the ratio 
    # R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_infty) 
    # is maximized by "peaky" signals.
    # For a delta function (one non-zero element), R = 1.
    # For a constant vector, R = 1/N = 0.02.
    # If the seed (constant) scores 1.0, the delta should score ~50.
    
    # We return a delta function centered or at the start.
    # The problem does not specify centering, only the sequence.
    # Autocorrelation is shift-invariant in terms of peak value for the ratio.
    h = np.zeros(N)
    h[0] = 1.0
    return list(h)
# EVOLVE-BLOCK-END
