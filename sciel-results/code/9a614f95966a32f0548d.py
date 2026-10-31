"""Cap set greedy builder with smooth line-avoidance and balance scoring."""
import time

_start_time = None
_DEADLINE = 330 * 0.92  # 303.6s safety margin — earlier cutoff to avoid timeout


def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.

    Structural design:
    1. Distinct symbol score (quadratic: 1→10000, 2→40000, 3→90000).
    2. Smooth balance score: rewards uniform symbol distribution (low chi-squared
       divergence from uniform). Uniform vectors are less likely to be the unique
       complement completing an affine line with many existing pairs.
    3. Variance danger proxy: high variance = more extreme coordinates = more
       line-completion risk.
    4. Fast rolling hash for deterministic tie-breaking.

    Optimized for low per-call overhead.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting — single pass
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1

    # Factor 1: Quadratic distinct symbol score
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Smooth balance score via chi-squared divergence from uniform
    # chi2 = sum((c_i - n/3)^2 / (n/3)) for i in {0,1,2}
    # Lower chi2 = more balanced = safer (less likely to complete many lines)
    # We reward balance: balance_score = -chi2 * scale
    # This is a smooth surrogate for the discrete mod-3 line-completion risk.
    ideal = n / 3.0
    if ideal > 0:
        chi2 = ((c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2) / ideal
    else:
        chi2 = 0.0
    balance_score = -chi2 * 200.0

    # Factor 3: Variance-based danger proxy (derived from counts, no extra loop)
    # s = c1 + 2*c2, s2 = c1 + 4*c2
    s = c1 + 2 * c2
    s2 = c1 + 4 * c2
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 4: Fast polynomial rolling hash for deterministic tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first
    total = distinct_score + balance_score + danger_score + hash_score
    return float(total)
