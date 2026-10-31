import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Smooth quadratic distinct-symbol score.
    2. Secondary: Spectral smoothness proxy (penalize high-frequency changes).
       We approximate spectral smoothness by computing the sum of squared 
       second-order differences. Vectors with low "curvature" are smoother.
    3. Tertiary: Quadratic balance penalty (L2 deviation from uniform).
    4. Quaternary: Symmetry breaking (favor 2s to explore full space).
    5. Quinary: Polynomial hash for tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    
    # First pass: counts and sums for balance/distinct
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4

    # 1. Distinct score (quadratic)
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # 2. Spectral Smoothness Proxy
    # Penalize rapid changes in symbol values. 
    # We compute sum of (v[i+2] - 2*v[i+1] + v[i])^2 for i in range(n-2).
    # This is a discrete Laplacian, which penalizes high-frequency components.
    # Lower curvature = smoother = better.
    curvature = 0.0
    if n >= 3:
        # Unroll loop for speed if possible, but n is variable
        # We use the vector directly
        for i in range(n - 2):
            d = vector[i + 2] - 2 * vector[i + 1] + vector[i]
            curvature += d * d
    else:
        # For small n, just use first-order differences or 0
        if n == 2:
            d = vector[1] - vector[0]
            curvature = d * d
        # n=1 curvature is 0
    
    # Normalize curvature by n to make it scale-invariant
    smoothness_score = -curvature * 100.0 / max(n, 1)

    # 3. Quadratic Balance Penalty
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # 4. Symmetry Breaking (favor 2s)
    weight_score = c2 * 100.0

    # 5. Hash for tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    total = distinct_score + smoothness_score + balance_score + weight_score + hash_score
    return float(total)
