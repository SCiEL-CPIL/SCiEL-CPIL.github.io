"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a binary error-correcting code of length 18 with minimum distance 6.
    Uses a systematic search with bit-parallel conflict masks to find a large code.
    """
    N = 18
    MIN_DIST = 6
    
    size = 1 << N
    forbidden = np.zeros(size, dtype=bool)
    
    # Precompute all masks of weight < MIN_DIST
    # Weight < 6 means weight 0 to 5.
    # There are sum_{i=0}^{5} C(18, i) = 12616 such masks.
    masks = [i for i in range(size) if i.bit_count() < MIN_DIST]
    masks = np.array(masks, dtype=np.uint32)
    
    # Function to mark forbidden neighbors
    def mark_forbidden(c):
        # c is an integer
        # neighbors = (c XOR masks) % size
        # forbidden[neighbors] = True
        neighbors = (c ^ masks) % size
        forbidden[neighbors] = True
        
    # Start with 0
    code = [0]
    mark_forbidden(0)
    
    # Iterate through candidates in Gray code order to ensure locality
    # Gray code ensures that consecutive numbers differ by only one bit,
    # which may help in finding clusters of valid codewords.
    
    # Generate Gray code sequence for N bits
    # Gray code i is i ^ (i >> 1)
    # We iterate through all 2^N Gray codes
    
    # Start with 0 (already added)
    # Next Gray code is 1 ^ (1 >> 1) = 1 ^ 0 = 1
    # Then 2 ^ (2 >> 1) = 2 ^ 1 = 3
    # Then 3 ^ (3 >> 1) = 3 ^ 1 = 2
    # etc.
    
    # We can just iterate i from 1 to size-1 and compute gray = i ^ (i >> 1)
    
    for i in range(1, size):
        c = i ^ (i >> 1)
        if not forbidden[c]:
            code.append(c)
            mark_forbidden(c)
            
    return code

# EVOLVE-BLOCK-END
