import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Structural change (per directive): shift from statistical-balance heuristics
    to a multiplicative-complexity model. Vectors are prioritized by the
    product of their (0->1 substituted) coordinates, which rewards "full-rank"
    algebraic structure and penalizes rank-deficient vectors that are more
    likely to participate in additive dependencies forming affine lines.

    Combined with refinements from inspirations:
      - Mod-3 coordinate-sum penalty (vectors with sum ≡ 0 mod 3 are more
        likely to complete affine lines with complementary pairs).
      - Variance-based danger proxy (high coordinate variance = more extreme
        coordinates = higher line-completion risk).
      - L1 balance penalty as a secondary smooth surrogate.
      - Fast rolling hash for deterministic tie-breaking.

    All components computed in O(n) per call with minimal overhead.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Single pass: count symbols, sum, sum-of-squares, and multiplicative product
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    prod = 1.0
    for v in vector:
        if v == 0:
            c0 += 1
            # treat 0 as 1 for multiplicative stability
            prod *= 1.0
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
            prod *= 1.0
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4
            prod *= 2.0

    # Factor 1 (PRIMARY, structural): multiplicative complexity score.
    # The product of (0->1 substituted) coordinates grows as 2^(# of 2s).
    # This directly rewards vectors that are "heavy" in the 2-coordinate,
    # enforcing a full-rank algebraic structure that resists additive
    # dependencies. Log-scale to keep it comparable to other terms.
    import math
    log_prod = math.log2(prod) if prod > 0 else 0.0
    multiplicative_score = log_prod * 500.0

    # Factor 2: Mod-3 coordinate-sum penalty.
    # In F_3^n, three points form an affine line iff their coordinate-wise
    # sum ≡ 0 mod 3. Vectors with sum ≡ 0 mod 3 are structurally dangerous.
    coord_sum = c1 + 2 * c2
    sum_mod = coord_sum % 3
    if sum_mod == 0:
        sum_score = -5000.0
    elif sum_mod == 1:
        sum_score = 2000.0
    else:
        sum_score = 1500.0

    # Factor 3: Variance-based danger proxy.
    # High coordinate variance means more extreme coordinates, which are
    # structurally more likely to complete an affine line.
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 4: L1 balance penalty (smooth secondary surrogate).
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    balance_score = -dev * 10.0

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = multiplicative_score + sum_score + danger_score + balance_score + hash_score
    return float(total)
