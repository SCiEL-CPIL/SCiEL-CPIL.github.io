"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Generate Thue-Morse sequence of length N
    # Thue-Morse: t(n) = (-1)^(popcount(n))
    # This sequence has good autocorrelation properties
    tm = np.empty(N, dtype=np.float64)
    for i in range(N):
        # Efficient popcount using bit manipulation
        x = i
        bits = 0
        while x:
            bits += x & 1
            x >>= 1
        tm[i] = 1.0 if (bits % 2 == 0) else -1.0
    return tm
# EVOLVE-BLOCK-END
