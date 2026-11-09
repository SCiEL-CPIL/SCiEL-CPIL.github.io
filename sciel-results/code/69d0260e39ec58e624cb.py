"""Rewritten program using MODEL-AND-OPTIMIZE approach with bitwise integer construction."""
import time

N = 100
DEADLINE_LIMIT = 145  # Safety margin from 150s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Model-and-Optimize approach for OneMax (100 bits).
    
    Model: The solution is represented as a 128-bit integer where the lower 100 bits
    correspond to the genotype. Each bit position i (0 <= i < 100) represents the
    value of the i-th character in the binary string.
    
    Objective: Maximize the count of ones $J = \sum_{i=0}^{N-1} b_i$.
    
    Optimization:
    The objective is linear and separable. The maximum value for each bit $b_i$ is 1.
    In the integer representation, this corresponds to setting all bits in the lower
    100 positions to 1. This is achieved by computing $2^{100} - 1$, which results
    in an integer with exactly 100 trailing 1-bits.
    
    This leverages arbitrary-precision integer arithmetic as a distinct computational
    pathway from string repetition. The value $2^{100} - 1$ is the unique integer
    whose binary representation (padded to 100 bits) consists entirely of '1's.
    
    Returns:
        A binary string of length N consisting entirely of '1's.
    """
    start_time = time.time()
    
    # The optimization converges immediately to the boundary of the parameter space
    # where all bits are 1.
    # Construct the integer representation: 2^100 - 1 has 100 ones in binary.
    optimal_int = (1 << N) - 1
    
    # Format the integer as a binary string of exactly N characters.
    # '0100b' ensures zero-padding to 100 bits, though the number already has 100 bits.
    result = format(optimal_int, f'0{N}b')
    
    # Explicit deadline guard (though O(1) solution makes this redundant, 
    # it satisfies the directive for explicit time tracking)
    if time.time() - start_time > DEADLINE_LIMIT:
        return '0' * N
        
    return result
# EVOLVE-BLOCK-END
