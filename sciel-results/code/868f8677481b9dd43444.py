"""Program for the third autocorrelation inequality task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct a signed step function using a Thue-Morse derived pattern
    to minimize the autoconvolution upper bound C3.
    
    The Thue-Morse sequence provides a high-contrast, low-autocorrelation
    pattern that exploits orthogonality of low-frequency components.
    """
    # Generate Thue-Morse sequence of length N
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    heights = np.ones(N, dtype=np.float64)
    for i in range(N):
        # Count number of 1 bits in i
        count = bin(i).count('1')
        if count % 2 == 1:
            heights[i] = -1.0
        else:
            heights[i] = 1.0
    
    return heights
# EVOLVE-BLOCK-END
