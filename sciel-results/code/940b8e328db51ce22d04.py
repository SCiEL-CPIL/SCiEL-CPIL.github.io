"""Cap set greedy builder with sum-of-squares geometric priority."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 325  # seconds, safety margin before 330s kill

def _is_cap_set_safe(vec, current_set):
    """Check if adding vec to current_set keeps it a cap set.
    A cap set has no three distinct points a,b,c with a+b+c ≡ 0 mod 3.
    """
    for a in current_set:
        for b in current_set:
            if a == b or b == vec or a == vec:
                continue
            # Check if a, b, vec form an affine line: a + b + vec ≡ 0 mod 3
            # i.e., vec ≡ -a-b mod 3
            diff = [(-a[i] - b[i] - vec[i]) % 3 for i in range(len(vec))]
            if all(d == 0 for d in diff):
                return False
    return True

def priority(vector, n):
    """Priority for greedy cap set construction.
    
    Uses sum of squares of coordinate values to break symmetry in the
    ternary grid. Vectors with more 1s and 2s (higher sum of squares)
    are favored because they occupy positions that are less prone to
    forming affine lines due to the non-linear distribution of lines
    in {0,1,2}^n.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -1e18
    
    v = tuple(vector)
    n = len(v)
    
    # Sum of squares: 0^2=0, 1^2=1, 2^2=4
    # This heavily favors 2s over 1s over 0s
    sum_sq = sum(x * x for x in v)
    
    # Normalize by n to get per-coordinate average
    ss_score = sum_sq / n  # ranges from 0 to 4
    
    # Count of each coordinate value for secondary heuristics
    c0 = sum(1 for x in v if x == 0)
    c1 = sum(1 for x in v if x == 1)
    c2 = sum(1 for x in v if x == 2)
    
    # Distinct values bonus: prefer using all three values
    distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    
    # Non-linear penalty for extreme imbalance
    # If all coordinates are the same value, that's very line-prone
    max_count = max(c0, c1, c2)
    imbalance_penalty = 0.2 * (max_count / n) ** 2
    
    # Sum of coordinates mod 3 - avoid sums that are 0 (more line-prone)
    coord_sum = sum(v) % 3
    sum_penalty = 0.05 if coord_sum == 0 else 0.0
    
    # Deterministic pseudo-random tie-breaker based on vector
    # Use a fast linear hash
    h = 0
    for i, x in enumerate(v):
        h = (h * 31 + x * (i + 1)) & 0xFFFFFFFF
    tie_break = (h / 0xFFFFFFFF) * 0.005
    
    total = ss_score + 0.1 * distinct - imbalance_penalty - sum_penalty + tie_break
    return total

# EVOLVE-BLOCK-END
