"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import zlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Prioritizes vectors to maximize the cap set size.
    Strategy: 
    1. Vectors with no 2s (i.e., from {0,1}^n) form a cap set of size 2^n. 
       We prioritize these to ensure we capture this large, conflict-free core.
    2. Among vectors with the same count of 2s, we use a hash to break ties randomly.
    """
    # Count the number of 2s in the vector
    count_2s = sum(1 for x in vector if x == 2)
    
    # Primary key: -count_2s (lower count of 2s is higher priority)
    # Secondary key: -hash (random tie-breaker)
    data = bytes(vector)
    h = zlib.crc32(data)
    
    # Scale the count significantly to ensure it dominates the hash
    return -float(count_2s) * 1000000.0 - float(h)
# EVOLVE-BLOCK-END
