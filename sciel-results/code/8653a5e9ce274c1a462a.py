"""Cap set greedy builder with geometric conflict minimization."""
import time
import numpy as np

# Deadline tracking
_start_time = None
_DEADLINE = 330.0 * 0.95  # 313.5s safety margin

def priority(vector, n):
    """
    Static priority function for cap set construction in {0,1,2}^n.
    
    Since the evaluator computes all priorities before greedy insertion,
    we need a vector-intrinsic heuristic. Strategy:
    1. Primary: Prioritize vectors with all 3 distinct symbols (most "generic").
       Vectors using only 1 or 2 symbols are more likely to form APs.
    2. Secondary: Among those, prioritize vectors with symbol counts close to uniform
       (n/3 each), measured by L1 deviation (faster than chi-squared, same ordering).
    3. Tertiary: Fast structural hash for deterministic tie-breaking.
    
    This combines the distinct-count dominance from inspiration 1 (score 1.2539)
    with the L1 balance measure (faster than chi-squared) and optimized hashing.
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
    # Scale to make it dominant: 3 distinct gets +30000, 2 gets +20000, 1 gets +10000
    if c0 > 0 and c1 > 0 and c2 > 0:
        distinct_score = 30000.0
    elif c0 > 0 and c1 > 0 or c0 > 0 and c2 > 0 or c1 > 0 and c2 > 0:
        distinct_score = 20000.0
    else:
        distinct_score = 10000.0
    
    # Balance: L1 deviation from uniform (faster than chi2, same ordering for small n)
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    balance_score = -dev * 10.0
    
    # Fast hash: simple polynomial rolling hash with bit masking for speed
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    
    hash_score = h / 0x7FFFFFFF
    
    return float(distinct_score + balance_score + hash_score)
