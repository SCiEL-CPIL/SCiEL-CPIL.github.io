import zlib
import time

_START = time.time()
_DEADLINE = _START + 280.0  # safety margin before 330s kill

def priority(vector, n):
    """
    Structural priority for cap set construction.
    
    Key insight: ANY subset of {0,2}^n is a cap set, since for a,b,c in {0,2}^n
    with a+b+c≡0 mod 3, each coordinate must sum to 0 mod 3, forcing a=b=c.
    So all 2^n vectors with only 0s and 2s are mutually compatible.
    
    Strategy:
    1. Vectors with no 1s get highest priority (they're always safe)
    2. Vectors with 1s: use exponential decay on count of 1s
    3. Deterministic hash for tie-breaking
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return 0.0
    
    c1 = vector.count(1)
    
    # Vectors with no 1s are always safe - give them priority > 1.0
    if c1 == 0:
        data = bytes(vector)
        crc = zlib.crc32(data)
        return 1.0 + float(crc) / 0xFFFFFFFF
    
    # For vectors with 1s, use exponential decay on c1
    # Fewer 1s = higher priority, smooth decay
    if n > 0:
        one_score = 2.718281828459045 ** (-0.5 * c1 / n)
    else:
        one_score = 1.0
    
    # Hash for deterministic tie-breaking
    data = bytes(vector)
    crc = zlib.crc32(data)
    hash_score = float(crc) / 0xFFFFFFFF
    
    # Combine: exponential decay dominates, hash for ties
    score = 0.9 * one_score + 0.1 * hash_score
    
    return score
