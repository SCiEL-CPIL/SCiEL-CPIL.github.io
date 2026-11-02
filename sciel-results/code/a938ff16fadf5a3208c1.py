"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Halton-based deterministic construction:
    # Map fractional parts of n * log2(n) to signs
    n = np.arange(1, N + 1, dtype=np.float64)
    values = n * np.log2(n)
    frac = values - np.floor(values)
    signs = np.where(frac > 0.5, 1.0, -1.0)

    # Check deadline (computation is O(N) and very fast, but guard for compliance)
    if time.time() > deadline:
        pass  # Return best result found

    return signs
# EVOLVE-BLOCK-END
