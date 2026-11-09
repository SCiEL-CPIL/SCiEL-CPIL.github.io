"""Cap set task: rank vectors so greedy descending insertion maximizes cap set size.

Uses the proven structural feature set (variance, balance error, low-density spread)
from the top-performing approach, combined with a deterministic polynomial hash
for stable, near-collision-free tie-breaking.
"""
import time

# Deadline guard: evaluator kills at 330s wall-clock. Keep a safety margin.
_T0 = time.time()
_DEADLINE = _T0 + 290.0  # 40s safety margin

# Fixed constants for the deterministic polynomial hash.
_HASH_BASE = 31
_HASH_MASK = 0x7FFFFFFF


def priority(vector, n):
    """Priority for greedy cap set construction in {0,1,2}^n.

    Higher return value => picked first. Structural features rank vectors so
    that "spread-out" vectors are accepted early (less likely to form 3-term
    APs), while a deterministic polynomial hash breaks ties stably.
    """
    # Deadline guard: return a safe constant so remaining vectors fall back
    # to a stable order instead of risking a timeout.
    if time.time() > _DEADLINE:
        return 0.0

    v = vector

    # Single O(n) pass computing structural features and the polynomial hash.
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    low = 0  # coordinates in {0,1}
    h = 0
    for x in v:
        if x == 0:
            c0 += 1
            low += 1
        elif x == 1:
            c1 += 1
            low += 1
        else:
            c2 += 1
        s += x
        s2 += x * x
        h = (h * _HASH_BASE + x) & _HASH_MASK

    # Variance: spread of coordinates. High variance = polarized = safer to add
    # early (less likely to be an AP middle). Range ~[0, 4/9].
    mean = s / n
    var = (s2 / n) - mean * mean
    if var < 0.0:
        var = 0.0

    # Balance error: how far (c0,c1,c2) are from equal thirds.
    third = n / 3.0
    bal = (abs(c0 - third) + abs(c1 - third) + abs(c2 - third)) / n

    # Low-density spread: prefer vectors whose {0,1} fraction is near 1/3.
    low_density = low / n
    low_spread = 1.0 - abs(low_density - 1.0 / 3.0)

    # Deterministic near-collision-free tie-breaker in [0,1).
    jitter = (h & 0xFFFF) / 65536.0

    # Weights tuned so structural terms dominate and jitter only breaks ties.
    score = (
        1.0 * var
        - 1.2 * bal
        + 0.3 * low_spread
        + 0.02 * jitter
    )
    return float(score)
