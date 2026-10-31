# EVOLVE-BLOCK-START
import time
import collections

_start_time = None
_deadline = 320.0  # safety margin before 330s kill

def priority(vector, n):
    """
    Static priority for cap set construction.
    
    Since the evaluator computes all priorities before greedy insertion,
    we use a vector-intrinsic heuristic: prefer vectors with balanced
    coordinate distributions (entropy-like), as these are less likely
    to form affine lines with many other vectors during greedy build.
    
    Fast per-call: use Counter for symbol counts, simple L1 balance.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if time.time() - _start_time > _deadline:
        return 0.0
    
    # Fast symbol counting
    counts = [0, 0, 0]
    for v in vector:
        counts[v] += 1
    
    c0, c1, c2 = counts
    
    # L1 distance from uniform distribution (n/3 each)
    # Lower = more balanced = better
    target = n / 3.0
    l1 = abs(c0 - target) + abs(c1 - target) + abs(c2 - target)
    
    # Penalize vectors with all same symbol (they form lines with many others)
    distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    all_same_penalty = 10.0 if distinct == 1 else 0.0
    
    # Prefer vectors with more distinct symbols (more "spread out")
    distinct_bonus = distinct * 0.1
    
    # Combined: minimize L1, avoid all-same, reward distinctness
    score = -l1 - all_same_penalty + distinct_bonus
    
    return score
# EVOLVE-BLOCK-END
