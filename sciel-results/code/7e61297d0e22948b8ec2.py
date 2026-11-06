import zlib
import time

_START = time.time()
_DEADLINE = _START + 280.0  # safety margin before 330s kill

def priority(vector, n):
    """
    Structural priority for cap set construction.
    
    Key insights:
    - A vector with many 1s is more likely to be the middle of a 3-AP
      (since for a,b,c with a+b+c≡0 mod 3, if b has coordinate 1,
      then a and c coordinates sum to 2 mod 3, i.e., {0,2} or {2,0})
    - Vectors that are "extremal" (mostly 0s and 2s, few 1s) are safer
    - Among safe vectors, prefer those with a specific pattern that
      minimizes pairwise conflict potential
    
    Strategy:
    1. Primary: count of 1s (fewer is better) - use exponential decay for smoothness
    2. Secondary: balance between 0s and 2s (prefer balanced binary vectors)
    3. Tertiary: deterministic hash for tie-breaking
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a low value to stop adding more vectors
        return 0.0
    
    # Count occurrences
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Primary signal: fewer 1s is better
    # Use exponential decay: c1=0 -> 1.0, decays smoothly as c1 increases
    # This provides a smooth surrogate for the hard constraint
    if n > 0:
        one_score = 2.718281828459045 ** (-0.5 * c1 / n)
    else:
        one_score = 1.0
    
    # Secondary signal: prefer balanced 0/2 vectors (maximize min(c0,c2) / (c0+c2))
    # Balanced vectors are more "spread out" and less likely to form 3-APs
    if c0 + c2 > 0:
        balance = 2.0 * min(c0, c2) / (c0 + c2)
    else:
        balance = 0.0
    
    # Tertiary: deterministic hash for stable tie-breaking
    data = bytes(vector)
    crc = zlib.crc32(data)
    hash_score = float(crc) / 0xFFFFFFFF
    
    # Combined score: weight primary heavily, then secondary, then hash
    # The all-1s vector is worst, balanced 0/2 vectors are good
    score = 0.7 * one_score + 0.2 * balance + 0.1 * hash_score
    
    return score
