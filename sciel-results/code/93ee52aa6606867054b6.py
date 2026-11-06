import time
import zlib
import sys
from collections import Counter

# Deadline guard: kill ourselves before the 330s wall-clock limit.
# We track elapsed time and if we get close, we return a safe fallback.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 10s safety margin

# Cache for computed priorities (per n)
_priority_cache = {}

def _check_deadline():
    """Return True if we should stop and return a fallback."""
    return (time.time() - _START_TIME) > _DEADLINE

def priority(vector, n):
    """
    Assigns a priority to a vector for greedy cap set construction.
    
    Strategy:
    1. Primary: Balance of symbol counts (0, 1, 2) - vectors closer to 
       equal distribution (n/3 each) get higher priority.
    2. Secondary: Spectral repulsion via hash-based approximation.
       We use a multi-level hash to create a pseudo-random but structured
       ordering that breaks ties in a way that encourages diversity.
    3. Tertiary: Stable hash for deterministic tie-breaking.
    
    The balance heuristic works because cap sets in F_3^n benefit from
    vectors that are "spread out" - having roughly equal counts of each
    symbol means the vector is far from the "diagonal" where all coordinates
    are equal (which creates many 3-term arithmetic progressions).
    """
    # Check deadline
    if _check_deadline():
        # Fallback: simple hash-based priority
        try:
            b = bytes(vector)
        except (TypeError, ValueError):
            b = str(vector).encode('utf-8')
        return float(zlib.crc32(b))
    
    # Get or compute priority
    cache_key = (n, vector)
    if cache_key in _priority_cache:
        return _priority_cache[cache_key]
    
    # Compute balance score
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    target = n / 3.0
    
    # Deviation from balanced counts - lower is better
    dev = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
    
    # We want high priority for low deviation, so invert
    balance_score = -dev
    
    # Add a spectral-like component using multiple hash levels
    # This creates structured randomness that helps break ties
    # in a way that promotes diversity in the selected set
    
    # Level 1: Fast CRC32 hash
    try:
        b = bytes(vector)
    except (TypeError, ValueError):
        b = str(vector).encode('utf-8')
    
    h1 = zlib.crc32(b)
    
    # Level 2: Different hash for decorrelation
    h2 = zlib.crc32(b, 0x12345678)
    
    # Level 3: Another variation
    h3 = zlib.crc32(b, 0x87654321)
    
    # Combine hashes into a small perturbation
    # The perturbation should be small relative to the balance score
    # to ensure balance is the primary driver
    perturbation = (h1 * 0.1 + h2 * 0.01 + h3 * 0.001) / 1e8
    
    # Also penalize vectors with many repeated values (heuristic:
    # such vectors may create more 3-APs with other vectors)
    max_count = max(c0, c1, c2)
    repeat_penalty = (max_count - target) ** 2 * 0.1
    
    final_score = balance_score - repeat_penalty + perturbation
    
    # Cache the result
    _priority_cache[cache_key] = final_score
    
    return final_score
