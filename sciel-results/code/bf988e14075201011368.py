"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a binary error-correcting code of length 18 with minimum distance 6.
    Uses a greedy algorithm with bit-parallel conflict masks and Gray-code ordering.
    """
    N = 18
    MIN_DIST = 6
    
    size = 1 << N
    forbidden = np.zeros(size, dtype=bool)
    
    # Precompute all masks of weight < MIN_DIST (i.e., weight 0 to 5)
    # There are sum_{i=0}^{5} C(18, i) = 12616 such masks.
    masks = np.array([i for i in range(size) if i.bit_count() < MIN_DIST], dtype=np.uint32)
    
    def mark_forbidden(c):
        neighbors = (c ^ masks) % size
        forbidden[neighbors] = True
    
    # Generate Gray code sequence for 18-bit integers
    # Gray code: g(i) = i ^ (i >> 1)
    # We iterate through all 2^18 values in Gray code order.
    # This ensures contiguous exploration in Hamming space.
    
    code = []
    
    # Process candidates in Gray code order
    # For each Gray code value, check if it's forbidden
    # If not, add to code and mark its neighbors as forbidden
    
    for i in range(size):
        c = i ^ (i >> 1)
        if not forbidden[c]:
            code.append(c)
            mark_forbidden(c)
    
    return code

# EVOLVE-BLOCK-END
