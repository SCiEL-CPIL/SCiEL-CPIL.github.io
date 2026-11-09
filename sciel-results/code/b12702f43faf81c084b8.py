"""Cap set greedy construction with structure-aware static priority."""
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
    2. Favor vectors with low "self-correlation" (fewer pairs of equal coordinates),
       as these are more 'generic' and less likely to form structured APs with
       other generic vectors.
    3. Use a robust hash for deterministic tie-breaking to avoid symmetry traps.
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # 1. Balance Penalty: Minimize deviation from target distribution
    #    Target: c0=n/6, c1=n/3, c2=n/2 (average digit 1.5)
    t0 = n / 6.0
    t1 = n / 3.0
    t2 = n / 2.0
    balance_penalty = (c0 - t0) ** 2 + (c1 - t1) ** 2 + (c2 - t2) ** 2
    
    # 2. Structure Penalty: Minimize number of equal coordinate pairs
    #    High number of equal pairs indicates the vector lies on a "diagonal"
    #    or has strong symmetry, which can be detrimental in cap set context.
    equal_pairs = 0
    # Optimization: Count frequencies and use combinatorics
    # freq[0] is c0, freq[1] is c1, freq[2] is c2
    # Number of pairs with same value is C(c0, 2) + C(c1, 2) + C(c2, 2)
    def comb2(k):
        return k * (k - 1) // 2
    
    equal_pairs = comb2(c0) + comb2(c1) + comb2(c2)
    
    # 3. Sum Deviation Penalty
    #    Prefer vectors where sum of digits is close to 1.5*n (average value 1.5)
    #    This favors vectors with more 1s and 2s, and fewer 0s.
    s = c0 * 0 + c1 * 1 + c2 * 2
    sum_penalty = (s - 1.5 * n) ** 2
    
    # 4. Tie-breaker: Deterministic pseudo-random hash
    #    This ensures that among vectors with similar structural stats,
    #    the selection order is stable but not easily exploited by symmetry.
    #    We use a simple polynomial hash.
    h = 0
    for i in range(n):
        h = (h * 31 + vector[i]) % (10**9 + 7)
    
    # Combine scores
    # We want to MAXIMIZE the return value.
    # Penalties are subtracted.
    
    # Weights:
    # Balance is the strongest signal for "good" cap set candidates.
    # Structure penalty is secondary.
    # Sum penalty is tertiary.
    # Hash is for exact ties.
    
    score = -balance_penalty * 1000.0 
    score -= equal_pairs * 50.0
    score -= sum_penalty * 50.0
    score += h * 1e-6  # Stronger perturbation for tie-breaking
    
    return score
