import zlib
import time

# EVOLVE-BLOCK-START
# Global state for deadline tracking
_start_time = None
_DEADLINE = 330.0 * 0.95  # 313.5s safety margin

def priority(vector, n):
    """
    Static priority function for cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Prioritize vectors with all 3 distinct symbols (0, 1, 2).
       These are 'generic' and less likely to form affine lines with other
       generic vectors compared to vectors with restricted symbol sets.
    2. Secondary: Among vectors with the same distinct count, prioritize
       those with symbol counts close to uniform (n/3 each).
       Measured by L1 deviation from the ideal uniform distribution.
    3. Tertiary: Fast deterministic hash for tie-breaking.
    
    This structural heuristic outperforms random ordering by biasing the
    greedy selection towards 'central' vectors in the cube, which tend to
    have fewer immediate conflicts in the affine space AG(n,3).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        # This should rarely be hit as the heuristic is O(n)
        return 0.0
    
    # Fast symbol counting
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
    
    # Distinct count score
    # 3 distinct symbols is best, 2 is middle, 1 is worst
    # We scale these significantly to ensure they dominate the balance term
    if c0 > 0 and c1 > 0 and c2 > 0:
        distinct_score = 30000.0
    elif (c0 > 0 and c1 > 0) or (c0 > 0 and c2 > 0) or (c1 > 0 and c2 > 0):
        distinct_score = 20000.0
    else:
        distinct_score = 10000.0
    
    # Balance score: L1 deviation from uniform
    # Ideal count for each symbol is n/3
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    # We want lower deviation, so we subtract it
    # Scale it so it's significant but secondary to distinct count
    balance_score = -dev * 10.0
    
    # Tie-breaking hash
    # Use CRC32 for a fast, well-distributed hash
    try:
        h = zlib.crc32(bytes(vector))
    except Exception:
        # Fallback to sum if bytes conversion fails
        h = sum(vector)
        
    # Normalize hash to [0, 1) to add a small perturbation
    hash_score = (h % 10000) / 10000.0
    
    return float(distinct_score + balance_score + hash_score)
# EVOLVE-BLOCK-END
