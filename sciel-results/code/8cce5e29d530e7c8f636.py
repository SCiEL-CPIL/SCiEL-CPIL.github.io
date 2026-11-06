"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import zlib

_start_time = None

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're running out of time, return a neutral priority
    elapsed = time.time() - _start_time
    if elapsed > 295:
        return 0.0
    
    num_ones = sum(1 for v in vector if v == 1)
    num_zeros = sum(1 for v in vector if v == 0)
    num_twos = sum(1 for v in vector if v == 2)
    
    # Strong penalty for 1s: a vector with k ones can be the middle term
    # in an AP in many coordinate-wise combinations
    # The "danger" scales quadratically with the number of 1s because
    # each pair of coordinates with 1s creates more AP opportunities
    score = -num_ones * num_ones * 5.0 - num_ones * 3.0
    
    # Bonus for vectors that are "balanced" between 0 and 2
    # These are less likely to form APs as endpoints
    balance = min(num_zeros, num_twos)
    score += balance * 0.5
    
    # Use zlib.crc32 as a deterministic tie-breaker for equal-count vectors
    # This ensures consistent ordering without being biased toward any particular pattern
    vec_bytes = bytes(vector)
    crc = zlib.crc32(vec_bytes) & 0xFFFFFFFF
    score += (crc % 1000) * 0.0001
    
    return score
# EVOLVE-BLOCK-END
