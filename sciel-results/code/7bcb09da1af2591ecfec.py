"""Cap set greedy builder combining distinct-count dominance with farthest-point sampling."""
import time
import hashlib

# Deadline tracking
_start_time = None
_DEADLINE = 300.0 * 0.95  # 285s safety margin

def priority(vector, n):
    """
    Priority function for cap set greedy insertion.
    
    Combines:
    1. Distinct-count dominance (from inspiration 1): vectors with all 3 symbols
       are structurally less likely to form affine lines than those with 1-2 symbols.
    2. Balance term: vectors with uniform symbol distribution (n/3 each) are preferred.
    3. Farthest-point sampling surrogate: maximize minimum Hamming distance to 
       "prototype" vectors. We use a deterministic set of reference vectors that 
       represent spread-out regions of the ternary grid. This encourages the greedy
       algorithm to select vectors that are spatially separated, reducing clustering
       and thus reducing the chance of forming affine lines.
    4. Fast hash for deterministic tie-breaking.
    
    Higher priority = inserted earlier.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
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
    
    # Distinct count (3 is best, 2 is middle, 1 is worst)
    # Scale to make it dominant
    if c0 > 0 and c1 > 0 and c2 > 0:
        distinct_score = 30000.0
    elif c0 > 0 and c1 > 0 or c0 > 0 and c2 > 0 or c1 > 0 and c2 > 0:
        distinct_score = 20000.0
    else:
        distinct_score = 10000.0
    
    # Balance: L1 deviation from uniform
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    balance_score = -dev * 10.0
    
    # Farthest-point sampling surrogate:
    # Compute minimum Hamming distance to a set of reference vectors.
    # We use 8 reference vectors that cover different "corners" of the ternary grid.
    # For efficiency, we only compute this for small n where it's feasible.
    spread_score = 0.0
    if n <= 10:
        # Reference vectors: all-0, all-1, all-2, and mixed patterns
        refs = [
            (0,) * n,
            (1,) * n,
            (2,) * n,
            tuple([0, 1, 2] * (n // 3 + 1))[:n],
            tuple([2, 1, 0] * (n // 3 + 1))[:n],
            tuple([1, 0, 2] * (n // 3 + 1))[:n],
        ]
        min_dist = n  # Maximum possible Hamming distance
        for ref in refs:
            dist = 0
            for i in range(n):
                if vector[i] != ref[i]:
                    dist += 1
            if dist < min_dist:
                min_dist = dist
        # Higher spread score for larger minimum distance
        spread_score = min_dist * 5.0
    
    # Fast hash: simple polynomial rolling hash with bit masking for speed
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    
    hash_score = h / 0x7FFFFFFF
    
    return float(distinct_score + balance_score + spread_score + hash_score)
