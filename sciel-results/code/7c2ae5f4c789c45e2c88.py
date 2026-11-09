"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Constructs a signed step function using the Thue-Morse sequence.
    The Thue-Morse sequence is a deterministic, low-discrepancy sequence
    that minimizes the autoconvolution energy by ensuring balanced
    positive and negative contributions across different scales.
    """
    # Generate the Thue-Morse sequence of length N
    # t_0 = 0
    # t_{2n} = t_n
    # t_{2n+1} = 1 - t_n
    # Equivalent to parity of popcount(i)
    
    # Efficient generation using the recursive property:
    # Start with [0]
    # Next block is [0, 1] (complement of [0])
    # Next is [0, 1, 1, 0]
    # Next is [0, 1, 1, 0, 1, 0, 0, 1]
    
    tm = [0]
    while len(tm) < N:
        # The next part of the sequence is the bitwise inverse of the current part
        tm.extend(1 - x for x in tm)
        
    # Truncate to N
    tm = tm[:N]
    
    # Map 0 to 1 and 1 to -1
    # This creates a high-contrast signed step function
    heights = np.where(np.array(tm) == 0, 1.0, -1.0)
    
    return heights
# EVOLVE-BLOCK-END
