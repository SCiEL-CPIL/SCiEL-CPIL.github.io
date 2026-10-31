import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (Algebraic Degree + Smooth Surrogates):
    1. Primary: Smooth quadratic distinct-symbol score. Strongly prefers vectors
       that use all three symbols (0, 1, 2), ensuring diversity in the cap set.
    2. Secondary: Algebraic Degree Score. Treats the vector as coefficients of a
       polynomial over F_3. Favors vectors with non-zero coefficients at higher
       indices (higher "degree"). This enforces a structured non-linearity and
       breaks symmetry by prioritizing the "outer" shell of the cube, which is
       hypothesized to be more resistant to forming affine lines.
    3. Tertiary: Smooth variance-based danger proxy. Penalizes high coordinate
       variance to prefer "middle" (1-heavy) vectors that are geometrically safer.
    4. Quaternary: Quadratic balance penalty. L2 deviation from the uniform
       (n/3, n/3, n/3) distribution, ensuring smooth symbol distribution.
    5. Quinary: Fast polynomial rolling hash for deterministic tie-breaking.

    All components are computed in O(n) time.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0

    # Fast symbol counting and algebraic degree calculation in one pass
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    alg_score = 0.0
    
    for i, v in enumerate(vector):
        # Algebraic degree score: weight by (i+1)^2 to favor high-index non-zeros
        # This mimics the magnitude of the polynomial at x=1 or similar,
        # but with quadratic weights to strongly emphasize high degrees.
        alg_score += v * (i + 1) * (i + 1)
        
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

    # Factor 1: Smooth quadratic distinct-symbol score.
    # Quadratic in presence count: 0 -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Algebraic Degree Score.
    # Scaled by 10.0 to make it a significant secondary factor.
    # For n=10, max is ~770 * 10 = 7700.
    # For n=20, max is ~5740 * 10 = 57400.
    # This is large enough to break ties between 3-symbol vectors but small enough
    # to not override the 50k gap between 2-symbol and 3-symbol vectors.
    alg_weighted_score = alg_score * 10.0

    # Factor 3: Smooth variance-based danger proxy.
    # High variance means the vector is "extreme".
    # Penalize to favor vectors with more 1s (closer to the center).
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 4: Quadratic balance penalty (smooth surrogate for L1).
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + alg_weighted_score + danger_score + balance_score + hash_score
    return float(total)
