"""Cap set task: rank vectors so greedy descending insertion maximizes cap set size."""
import time
import math

# Explicit deadline guard: evaluator kills at 330s wall-clock.
_T0 = time.time()
_DEADLINE = _T0 + 285.0  # 45s safety margin

# Deterministic seed so the ranking is stable across the 3^n calls.
_SEED = 123456789


def _xorshift(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 7)
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF


def priority(vector, n):
    # Deadline guard: if we're running long, return a safe constant so the
    # remaining (tie) vectors fall back to lexicographic order.
    if time.time() > _DEADLINE:
        return 0.0

    v = vector
    # Fast path for the all-zero vector (always first in lex order, harmless).
    # Compute cheap O(n) structural features.
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    low = 0  # count of coordinates in {0,1}
    # Also compute pairwise "collision" features: count of (i,j) where v[i]==v[j]
    # This captures clustering which hurts cap set property
    # For efficiency, we can compute this from counts: sum_{a} c_a*(c_a-1)
    for x in v:
        if x == 0:
            c0 += 1
        elif x == 1:
            c1 += 1
            low += 1
        else:
            c2 += 1
            low += 1
        s += x
        s2 += x * x

    mean = s / n
    var = (s2 / n) - mean * mean
    if var < 0:
        var = 0.0

    # Balance: how close the (c0,c1,c2) counts are to equal thirds.
    third = n / 3.0
    bal = (abs(c0 - third) + abs(c1 - third) + abs(c2 - third)) / n

    # Low-coordinate density (0 or 1): cap sets often avoid clustering on 0/1.
    low_density = low / n

    # Pairwise same-value count: sum c_a*(c_a-1) for a in {0,1,2}
    # This measures how "clumped" the vector is. Lower is better for cap sets.
    pair_same = c0 * (c0 - 1) + c1 * (c1 - 1) + c2 * (c2 - 1)
    # Normalize by n*(n-1)
    if n > 1:
        pair_same_norm = pair_same / (n * (n - 1))
    else:
        pair_same_norm = 0.0

    # Deterministic pseudo-random component keyed by the vector, stable per call.
    h = _SEED
    for x in v:
        h = _xorshift(h ^ (x + 1) * 0x9E3779B97F4A7C15)
    rnd = (h & 0xFFFFFFFF) / 4294967296.0

    # Combine: prefer high variance (spread), low balance-error, moderate low_density,
    # low pair_same_norm (less clumping), plus a small deterministic tie-breaker.
    score = (
        1.0 * var
        - 1.2 * bal
        + 0.3 * (1.0 - abs(low_density - 1.0 / 3.0))
        - 0.5 * pair_same_norm
        + 0.05 * rnd
    )
    return float(score)
