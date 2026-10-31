"""Cap set greedy builder with quadratic presence and mod-3 line-avoidance priority."""
import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.

    Structural design:
    1. Quadratic symbol-presence score (smooth landscape: all-3 > two > one).
    2. Mod-3 coordinate-sum penalty (vectors with sum ≡ 0 mod 3 are more likely
       to complete affine lines with complementary pairs).
    3. Quadratic balance penalty (deviation from uniform coordinate distribution).
    4. Fast rolling hash for deterministic tie-breaking.

    The quadratic presence term creates a smooth ordering where vectors using
    all three symbols are strongly preferred, but the penalty for missing a
    symbol is graded rather than abrupt, giving the greedy process more
    informative intermediate ordering signals.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)

    # Factor 1: Quadratic presence score (smooth distinct-symbol reward)
    p0 = 1.0 if c0 > 0 else 0.0
    p1 = 1.0 if c1 > 0 else 0.0
    p2 = 1.0 if c2 > 0 else 0.0
    num_distinct = p0 + p1 + p2
    # Quadratic: 0→0, 1→10000, 2→40000, 3→90000
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Mod-3 sum penalty
    # In F_3^n, three points form an affine line iff their coordinate-wise sum ≡ 0 mod 3.
    # Vectors with sum ≡ 0 mod 3 are "dangerous" — they pair with many others to form lines.
    # Vectors with sum ≡ 1 or 2 mod 3 are safer.
    coord_sum = c0 * 0 + c1 * 1 + c2 * 2
    sum_mod = coord_sum % 3
    # Strong penalty for sum_mod == 0, mild preference for 1 over 2
    if sum_mod == 0:
        sum_score = -5000.0
    elif sum_mod == 1:
        sum_score = 2000.0
    else:
        sum_score = 1500.0

    # Factor 3: Quadratic balance penalty (deviation from uniform n/3)
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Fast polynomial rolling hash for deterministic tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF  # [0, 1) small tiebreaker

    # Combined priority: higher is added first
    total = distinct_score + sum_score + balance_score + hash_score
    return float(total)
