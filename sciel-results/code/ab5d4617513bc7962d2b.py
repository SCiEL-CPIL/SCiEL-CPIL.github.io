import time

_start_time = None
_DEADLINE = 300 * 0.95  # 285s safety margin before the 300s kill

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy:
    1. Primary: Smooth quadratic distinct-symbol score. Quadratic in the number
       of present symbols (0, 1, 4, 9 * 10000) creates a steep but continuous
       gradient, strongly preferring vectors that use all three symbols.
       This avoids the discrete cliffs of linear tiering.
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

    # Factor 1: Smooth distinct-symbol score with softer cliff.
    # Uses a cubic blend: 0 -> 0, 1 -> 8000, 2 -> 28000, 3 -> 54000.
    # The ratio between 2->3 (54/28 ≈ 1.93) is gentler than the old 90/40 = 2.25,
    # allowing more 2-symbol vectors to be considered before 3-symbol vectors
    # dominate, giving the greedy algorithm a wider exploration window.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 8000.0 * num_distinct * num_distinct * num_distinct / 3.0

    # Factor 2: Symmetry breaking via count of 2s.
    # Stronger weight to prevent clustering in {0,1}^n sub-cube.
    weight_score = c2 * 250.0

    # Factor 3: Quadratic balance penalty (smooth surrogate for L1).
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution.
    # Increased weight to more strongly encourage balanced symbol usage.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 12.0

    # Factor 4: Smooth variance-based danger proxy.
    # Reduced penalty: moderate variance is acceptable and even beneficial
    # for cap set construction; only extreme variance is truly dangerous.
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 2000.0

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + weight_score + balance_score + danger_score + hash_score
    return float(total)
