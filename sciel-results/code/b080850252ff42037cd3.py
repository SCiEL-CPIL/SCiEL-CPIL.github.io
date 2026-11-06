import time
import os
import hashlib

# Track wall-clock time for timeout
_start_time = time.time()
_TIMEOUT = 315.0  # Safety margin before 320s hard limit

# Precompute a deterministic salt for tie-breaking based on PID/hostname
# This ensures distinctness but deterministic per run
_salt = os.getpid() ^ (hash(os.uname().nodename) % (2**16))

def priority(vector, n):
    # Check timeout
    if time.time() - _start_time > _TIMEOUT:
        return -1e9

    # Fast counting
    c0 = 0
    c1 = 0
    c2 = 0
    # Use a faster hash than polynomial base-31 mod large prime
    # We want a 64-bit-ish hash for tie-breaking
    h = 0
    
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
        # Fast mixing hash (SplitMix64 style)
        h ^= v + 0x9e3779b97f4a7c15 + (h << 6) + (h >> 2)
        h &= 0xFFFFFFFFFFFFFFFF

    ideal = n / 3.0
    # Quadratic deviation from ideal balance
    dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # Penalize vectors that are too "concentrated" in any single symbol
    # This helps avoid vectors that are likely to form arithmetic progressions
    max_count = max(c0, c1, c2)
    concentration_penalty = max_count * max_count / n if n > 0 else 0
    
    # Small bonus for having all three symbols present (more diverse)
    diversity_bonus = 0.0
    if c0 > 0 and c1 > 0 and c2 > 0:
        diversity_bonus = 0.5

    # Combine: minimize dev and concentration, maximize diversity
    # Use the hash as a very small tie-breaker to ensure distinct priorities
    # This avoids the "lexicographic" feel while keeping determinism
    score = -dev - 0.1 * concentration_penalty + diversity_bonus - (h / (2**64)) * 1e-6

    return score
