"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Hypothesis: "C5" implies shift of 5.
    # Minimize overlap at shift 5 by maximizing correlation at shift 5.
    # Use a periodic binary sequence with period 10 (5 ones, 5 zeros) 
    # or period 5 (if sum allows, but sum must be 50).
    # Period 10: 5 ones, 5 zeros. Sum = 5 * (100/10) = 50.
    # h_i = 1 if (i % 10) < 5 else 0.
    
    h = np.zeros(N)
    for i in range(N):
        if (i % 10) < 5:
            h[i] = 1.0
        else:
            h[i] = 0.0
            
    # Check sum
    assert np.sum(h) == 50.0, f"Sum is {np.sum(h)}"
    
    return h
# EVOLVE-BLOCK-END
