"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_start_time = None

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're running out of time, return a neutral priority
    elapsed = time.time() - _start_time
    if elapsed > 300:
        return 0.0
    
    # Count the number of 1s in the vector
    # Vectors with fewer 1s are less likely to be middle terms of 3-term APs
    # so they are "safer" to include early in greedy construction
    num_ones = sum(1 for v in vector if v == 1)
    num_zeros = sum(1 for v in vector if v == 0)
    num_twos = sum(1 for v in vector if v == 2)
    
    # Primary: minimize number of 1s (they create AP opportunities)
    # Secondary: prefer balance between 0s and 2s (more "spread")
    # Tertiary: slight preference for vectors with more distinct values
    
    # The number of 1s is the dominant factor
    # A vector with k ones has C(k,1) ways to be a middle term in a coordinate
    # So we heavily penalize 1s
    
    # Use a strong negative weight for 1s
    score = -num_ones * 10.0
    
    # Small bonus for having both 0s and 2s (more spread)
    if num_zeros > 0 and num_twos > 0:
        score += 1.0
    
    # Very small tie-breaker: prefer vectors with more 0s than 2s (arbitrary but consistent)
    score += (num_zeros - num_twos) * 0.01
    
    return score
# EVOLVE-BLOCK-END
