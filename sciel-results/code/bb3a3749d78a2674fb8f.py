"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Structural change: Use a fixed pseudo-random signed sequence.
    # This replaces the constant 'ones' seed with a high-frequency signed function.
    # Random signs significantly reduce autocorrelation sidelobes compared to a DC signal.
    # A fixed seed ensures reproducibility and avoids the risk of a bad random draw.
    rng = np.random.default_rng(42)
    return rng.choice([-1.0, 1.0], size=N)
# EVOLVE-BLOCK-END
