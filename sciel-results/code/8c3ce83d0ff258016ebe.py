"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """Construct a signed step function with 400 equal intervals on [-1/4,1/4].
    
    Returns a sequence of 400 signed step heights in {-1, 1} that minimizes
    the autoconvolution upper bound C3.
    """
    # Use a deterministic pseudo-random ±1 sequence with good autocorrelation properties.
    # A high-quality PRNG produces sequences with near-orthogonal autocorrelation.
    rng = np.random.default_rng(42)
    x = rng.choice([-1, 1], size=N)
    
    return x
# EVOLVE-BLOCK-END
