import time
import math

_start_time = None
_DEADLINE = 310.0  # Safety margin before 330s kill

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Prioritize vectors with 3 distinct symbols.
    2. Among those, maximize entropy of symbol distribution (measures balance).
    3. Add pairwise disagreement score for local mixing.
    4. Use a fast linear hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Fast count of symbols
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
    
    # Distinct count
    distinct_count = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    
    # Entropy-based balance score (higher entropy = more balanced)
    # H = -sum(p_i * log(p_i)), max is log(3) when perfectly uniform
    if n > 0:
        p0 = c0 / n if c0 > 0 else 0.0
        p1 = c1 / n if c1 > 0 else 0.0
        p2 = c2 / n if c2 > 0 else 0.0
        entropy = 0.0
        if p0 > 0:
            entropy -= p0 * math.log(p0)
        if p1 > 0:
            entropy -= p1 * math.log(p1)
        if p2 > 0:
            entropy -= p2 * math.log(p2)
        # Normalize: 0 to 1 (1 = perfectly uniform)
        max_entropy = math.log(3.0)
        balance_score = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        balance_score = 0.0
    
    # Pairwise disagreement: count positions where adjacent pairs differ
    # This encourages local mixing and reduces affine line risk
    disagreement = 0
    for i in range(n - 1):
        if vector[i] != vector[i + 1]:
            disagreement += 1
    disagreement_score = disagreement / max(n - 1, 1)
    
    # Fast linear hash for tie-breaking
    h = 0
    base = 31
    for i, val in enumerate(vector):
        h = (h * base + val + i * 7) % (10**9 + 7)
    
    hash_score = h / (10**9 + 7)
    
    # Combined score:
    # - distinct_count dominates (3 > 2 > 1)
    # - entropy balance (higher = more balanced = better cap set property)
    # - pairwise disagreement (higher = more local mixing)
    # - hash breaks ties deterministically
    distinct_score = distinct_count * 10000.0
    balance_scaled = balance_score * 50.0
    disagreement_scaled = disagreement_score * 20.0
    
    return float(distinct_score + balance_scaled + disagreement_scaled + hash_score)
