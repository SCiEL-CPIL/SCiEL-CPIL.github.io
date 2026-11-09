"""Cap set greedy construction with balanced-digit priority and hash-based tie-breaking."""
import zlib
import time

# EVOLVE-BLOCK-START

_DEADLINE_SECONDS = 320.0
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.monotonic()
    return time.monotonic() - _start_time

def priority(vector, n):
    """
    Returns a priority score for a vector in {0,1,2}^n.
    Higher score means the vector is added earlier in the greedy process.
    
    Strategy:
    1. Primary key: Balance of digit counts (0, 1, 2).
       We prefer vectors with counts close to n/3. This is equivalent to minimizing
       the variance of the counts.
    2. Secondary key: A hash of the vector to break ties randomly.
       This avoids the structural biases of lexicographic ordering, which can
       lead to suboptimal packing in the greedy algorithm.
    """
    # Count digits
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Calculate variance of counts
    expected = n / 3.0
    var = (c0 - expected) ** 2 + (c1 - expected) ** 2 + (c2 - expected) ** 2
    
    # Primary score: negative variance (higher is better/more balanced)
    balance_score = -var
    
    # Secondary score: Hash for tie-breaking
    # Convert vector to bytes for hashing. Values are 0, 1, 2 so they fit in bytes.
    # This is faster than lex rank calculation and provides better distribution.
    b = bytes(vector)
    h = zlib.crc32(b)
    
    # Normalize hash to [0, 1)
    hash_score = h / 4294967296.0  # 2^32
    
    # Combine scores.
    # The hash term is scaled down significantly so it only affects the order
    # when balance_score is exactly equal (or within floating point noise, 
    # though integer counts make var exact for same count tuples).
    # We subtract hash_score to give priority to lower hash values (arbitrary choice).
    return balance_score - hash_score * 1e-6

# EVOLVE-BLOCK-END
