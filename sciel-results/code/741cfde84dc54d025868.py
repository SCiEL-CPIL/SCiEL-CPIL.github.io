"""Cap set greedy builder: quadratic presence, mod-3 line-avoidance, symmetry-breaking, and variance danger."""
import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.

    Design (combined from current + strongest inspirations):
    1. Quadratic symbol-presence score: smooth landscape favoring vectors
       that use all three symbols (0->0, 1->10000, 2->40000, 3->90000).
    2. Mod-3 coordinate-sum penalty: vectors with sum ≡ 0 mod 3 are more
       likely to complete affine lines; penalize strongly.
    3. Hamming-weight symmetry-breaking: favor vectors with more 2s to
       prevent greedy clustering in the {0,1}^n sub-cube and force
       uniform exploration of the full {0,1,2}^n space.
    4. Quadratic balance penalty: L2 deviation from uniform (n/3,n/3,n/3).
    5. Variance-based danger proxy: high coordinate variance means more
       extreme coordinates, structurally more likely to form lines.
    6. Fast rolling hash for deterministic tie-breaking.

    All components computed in O(n) with a single pass over the vector.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Single-pass symbol counting + sum + sum-of-squares
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

    # Factor 1: Quadratic distinct-symbol score
    # 0 distinct -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Mod-3 sum penalty
    # coord_sum = c1 + 2*c2 (since 0 contributes nothing)
    coord_sum = c1 + 2 * c2
    sum_mod = coord_sum % 3
    if sum_mod == 0:
        sum_score = -5000.0
    elif sum_mod == 1:
        sum_score = 2000.0
    else:
        sum_score = 1500.0

    # Factor 3: Hamming-weight symmetry-breaking (favor 2s)
    # Prevents clustering in {0,1}^n; pushes exploration into full space
    weight_score = c2 * 100.0

    # Factor 4: Quadratic balance penalty (deviation from uniform n/3)
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 5: Variance-based danger proxy
    # High variance = more extreme coordinates = higher line-completion risk
    if n > 0:
        mean = s / n
        variance = s2 / n - mean * mean
        danger_score = -variance * 5000.0
    else:
        danger_score = 0.0

    # Factor 6: Fast polynomial rolling hash for deterministic tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF  # [0, 1) small tiebreaker

    # Combined priority: higher is added first
    total = distinct_score + sum_score + weight_score + balance_score + danger_score + hash_score
    return float(total)
