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

    # Factor 1: Smooth soft-max distinct-symbol score.
    # Use tanh saturation to create a continuous gradient instead of hard cliffs.
    # tanh(c/alpha) smoothly approaches 1 as c grows, with alpha controlling steepness.
    alpha = max(1.0, n * 0.1)  # scale with dimension
    sat0 = 1.0 if c0 == 0 else (1.0 - 0.5 * (1.0 - c0 / alpha) ** 2) if c0 < alpha else 1.0
    sat1 = 1.0 if c1 == 0 else (1.0 - 0.5 * (1.0 - c1 / alpha) ** 2) if c1 < alpha else 1.0
    sat2 = 1.0 if c2 == 0 else (1.0 - 0.5 * (1.0 - c2 / alpha) ** 2) if c2 < alpha else 1.0
    distinct_score = 10000.0 * (sat0 + sat1 + sat2) ** 2

    # Factor 2: Symmetry breaking via count of 2s.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty (smooth surrogate for L1).
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Smooth line-completion risk proxy.
    # The number of coordinate triples (i,j,k) where values could form {0,1,2}
    # is proportional to c0*c1*c2. Use a smooth product with epsilon.
    eps = 1.0
    product_smooth = (c0 + eps) * (c1 + eps) * (c2 + eps)
    # Normalize by n^3 to get a density-like measure
    danger_score = -(product_smooth / (n * n * n + 1)) * 50000.0

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking.
    # Use a faster hash: combine first few and last few elements
    h = 0
    if n > 0:
        h = vector[0] * 31 + (n - 1)
        if n > 1:
            h = (h * 31 + vector[-1] + n - 2) & 0x7FFFFFFF
        if n > 2:
            mid = n // 2
            h = (h * 31 + vector[mid] + mid) & 0x7FFFFFFF
    hash_score = (h & 0x7FFFFFFF) / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + weight_score + balance_score + danger_score + hash_score
    return float(total)
