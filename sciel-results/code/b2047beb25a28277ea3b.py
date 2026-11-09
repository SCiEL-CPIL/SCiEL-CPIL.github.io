"""Cap set greedy construction with 2-gram entropy and structure-aware priority."""
import time
import math
import random

_START_TIME = time.time()
TIME_LIMIT = 280  # seconds

def priority(vector, n):
    """
    Static priority function for greedy cap set construction.
    
    Strategy:
    1. Favor vectors with balanced digit distribution (0, 1, 2 counts close to n/3).
    2. Favor vectors with high 2-gram entropy (diverse adjacent coordinate pairs),
       as these are more 'generic' and less likely to form structured APs.
    3. Penalize vectors with many equal-coordinate pairs (diagonal/symmetry).
    4. Use a robust hash for deterministic tie-breaking to avoid symmetry traps.
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    
    # 1. Balance Penalty: Minimize deviation from uniform distribution
    balance_penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # 2. 2-gram Entropy: Measure diversity of adjacent coordinate pairs
    #    High entropy means the vector has many different (v[i], v[i+1]) patterns,
    #    making it less likely to form APs with other vectors.
    if n >= 2:
        from collections import Counter
        pairs = [(vector[i], vector[i+1]) for i in range(n-1)]
        pair_counts = Counter(pairs)
        total_pairs = len(pairs)
        entropy = 0.0
        for count in pair_counts.values():
            p = count / total_pairs
            if p > 0:
                entropy -= p * math.log(p)
        # Normalize entropy to [0, 1] by dividing by max possible (log of #distinct pairs)
        # Max distinct pairs is 9 (all combinations of {0,1,2}x{0,1,2})
        max_entropy = math.log(min(total_pairs, 9))
        if max_entropy > 0:
            entropy_norm = entropy / max_entropy
        else:
            entropy_norm = 0.0
    else:
        entropy_norm = 0.0
    
    # 3. Structure Penalty: Minimize number of equal coordinate pairs
    def comb2(k):
        return k * (k - 1) // 2
    
    equal_pairs = comb2(c0) + comb2(c1) + comb2(c2)
    
    # 4. Sum Deviation Penalty
    s = c1 + 2 * c2
    sum_penalty = (s - n) ** 2
    
    # 5. Tie-breaker: Deterministic pseudo-random hash
    h = 0
    for i in range(n):
        h = (h * 31 + vector[i]) % (10**9 + 7)
    
    # Combine scores
    # We want to MAXIMIZE the return value.
    # 
    # Weight hierarchy:
    # - Balance penalty dominates (1000x) to establish clear tiers
    # - 2-gram entropy is secondary (200x) - rewards diverse local patterns
    # - Structure penalty is tertiary (50x)
    # - Sum penalty is quaternary (10x)
    # - Hash is for exact ties (1e-9)
    
    score = -balance_penalty * 1000.0
    score += entropy_norm * 200.0  # Favor high 2-gram entropy
    score -= equal_pairs * 50.0
    score -= sum_penalty * 10.0
    score += h * 1e-9  # Tiny perturbation for tie-breaking
    
    return score
