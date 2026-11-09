"""OneMax 100-bit binary construction: maximize number of ones."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    Uses arbitrary-precision integer arithmetic to generate the optimal all-ones genotype.
    """
    # The optimal solution for OneMax is all ones.
    # We represent this as an integer with the lower 100 bits set to 1.
    # (1 << 100) - 1 creates a number with 100 ones in binary.
    optimal_int = (1 << N) - 1
    
    # Format the integer as a binary string of length N, zero-padded on the left.
    # Since the integer has exactly N bits set to 1, the padding will be empty,
    # but '0100b' ensures the format is correct if N were smaller or bits were 0.
    result = format(optimal_int, f'0{N}b')
    
    return result
# EVOLVE-BLOCK-END
