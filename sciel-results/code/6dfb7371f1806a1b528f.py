"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Use a deterministic ±1 sequence with low autocorrelation
    # Generate using a fixed-seed pseudo-random sequence for reproducibility
    rng = np.random.default_rng(42)
    h = rng.choice([-1.0, 1.0], size=N)
    return h
# EVOLVE-BLOCK-END
