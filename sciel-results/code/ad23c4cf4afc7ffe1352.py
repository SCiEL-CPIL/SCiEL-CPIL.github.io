import time

_start_time = None
_DEADLINE = 310.0  # Safety margin before 330s kill

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Prioritize vectors with 3 distinct symbols.
    2. Among those, prioritize vectors with symbol counts close to uniform (n/3).
    3. Use a fast linear hash for deterministic tie-breaking.
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
    
    # Balance score: chi-squared statistic (lower is more balanced)
    ideal = n / 3.0
    if ideal > 0:
        chi2 = ((c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2) / ideal
    else:
        chi2 = 0.0
    
    # Fast linear hash for tie-breaking
    h = 0
    base = 31
    for i, val in enumerate(vector):
        h = (h * base + val + i * 7) % (10**9 + 7)
    
    hash_score = h / (10**9 + 7)
    
    # Combined score:
    # - distinct_count dominates (3 > 2 > 1)
    # - chi2 penalizes imbalance (lower chi2 = higher priority)
    # - hash breaks ties deterministically
    distinct_score = distinct_count * 10000.0
    balance_scaled = -chi2 * 10.0
    
    return float(distinct_score + balance_scaled + hash_score)
