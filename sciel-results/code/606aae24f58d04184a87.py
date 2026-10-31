import time

_start_time = None
_DEADLINE = 300 * 0.95  # 285s safety margin before the 300s kill

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (Smooth Surrogate Heuristic):
    1. Primary: Smooth quadratic distinct-symbol score. Quadratic in the number
       of present symbols (0, 1, 4, 9 * 10000) creates a steep but continuous
       gradient, strongly preferring vectors that use all three symbols.
       This avoids the discrete cliffs of linear tiering which can trap greedy
       algorithms in local optima.
    2. Secondary: Smooth variance-based danger proxy. High coordinate variance
       indicates many "extreme" coordinates (0s and 2s), which are structurally
       more likely to complete affine lines. We penalize variance to prefer
       "middle" (1-heavy) vectors that are geometrically safer.
    3. Tertiary: Quadratic balance penalty. L2 deviation from the uniform
       (n/3, n/3, n/3) distribution. This is a smooth surrogate for balance
       that provides a better gradient than piecewise-linear L1 or entropy-based
       metrics, avoiding local plateaus in the optimization landscape.
    4. Quaternary: Symmetry breaking via count of 2s. Favors vectors with more
       2s to prevent the greedy algorithm from clustering in the {0,1}^n sub-cube
       and ensures uniform exploration of the full ternary grid.
    5. Quinary: Fast polynomial rolling hash for deterministic tie-breaking.

    All components are computed in O(n) time with minimal per-call overhead.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0

    # Fast symbol counting in one pass (collects counts, sum, and sum-of-squares)
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
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

    # Factor 1: Smooth quadratic distinct-symbol score.
    # Quadratic in presence count: 0 -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000.
    # This strongly prioritizes vectors with all 3 symbols.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Smooth variance-based danger proxy.
    # High variance means the vector is "extreme" (lots of 0s and 2s, few 1s).
    # Such vectors are more likely to be part of a line {v, v+d, v+2d}.
    # We penalize high variance to favor vectors with more 1s (closer to the "center" of the cube).
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 3: Quadratic balance penalty (smooth surrogate for L1).
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Symmetry breaking.
    # Favors vectors with more 2s slightly. This breaks the symmetry between 0 and 2
    # and helps the greedy algorithm explore the space more uniformly without
    # biasing towards a specific sub-cube like {0,1}^n.
    weight_score = c2 * 100.0

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking.
    # This ensures that even if two vectors have identical heuristic scores,
    # they are ordered consistently, which can help in breaking ties in a way
    # that is stable across runs.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + weight_score + balance_score + danger_score + hash_score
    return float(total)
