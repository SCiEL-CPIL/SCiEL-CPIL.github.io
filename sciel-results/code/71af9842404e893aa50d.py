"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import zlib

def priority(vector, n):
    """
    Assign priority to vectors in {0,1,2}^n to maximize cap set size via greedy insertion.
    
    Strategy:
    - Use a pseudo-random priority to avoid structural traps in greedy selection.
    - The previous heuristic (prioritizing vectors with few 1s) leads to selecting the entire {0,2}^n set first,
      which is a maximal cap set but not the maximum size. This prevents adding many other vectors.
    - A pseudo-random order allows the greedy algorithm to explore a more diverse set of vectors,
      potentially finding a larger cap set closer to the theoretical maximum.
    """
    # Convert vector to a string representation for hashing
    # Using CRC32 for a fast, deterministic pseudo-random priority
    vec_str = ''.join(map(str, vector))
    return float(zlib.crc32(vec_str.encode()))
# EVOLVE-BLOCK-END
