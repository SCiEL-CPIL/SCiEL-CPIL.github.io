import time
import math
import random
from itertools import product

_START_TIME = time.time()
TIME_LIMIT = 275
_DEADLINE = _START_TIME + TIME_LIMIT

# Precompute position weights that favor zeros in "structurally critical" positions.
# In a 3-term AP (a, b, c) with a + c = 2b in F_3^n, the middle coordinate b_i
# is the "pivot". If b_i = 0, then a_i + c_i = 0, so c_i = -a_i = (3 - a_i) mod 3.
# This is a specific constraint. But more importantly, having 0 in a coordinate
# means that coordinate contributes 0 to any linear combination, reducing the
# "surface area" for collisions.
#
# Strategy: Give higher priority to vectors that have zeros in positions with
# high "criticality". We use a fixed weight vector that varies by position index,
# with a deterministic but non-trivial pattern.
#
# We also add a small time-varying perturbation to avoid getting stuck in
# a single local optimum, but the core structure is the zero-bias.

def _position_weights(n):
    """Generate position weights that favor zeros at certain indices."""
    weights = []
    for i in range(n):
        # Use a combination of:
        # - Higher weight for "central" positions (they appear in more APs as middle)
        # - A deterministic pseudo-random component for diversity
        center_dist = abs(i - n / 2.0) / max(n / 2.0, 1)
        # Inverse: closer to center = higher weight
        center_w = 1.0 - center_dist
        # Deterministic hash-based component
        h = (i * 2654435761) & 0xFFFFFFFF
        rand_w = (h / 4294967295.0)
        weights.append(0.6 * center_w + 0.4 * rand_w)
    return weights

# Precompute for common n values (up to ~15)
_weight_cache = {}

def _get_weights(n):
    if n not in _weight_cache:
        _weight_cache[n] = _position_weights(n)
    return _weight_cache[n]

def priority(vector, n):
    # Deadline guard
    if time.time() > _DEADLINE:
        return 0.0

    weights = _get_weights(n)

    # 1. Zero-bias score: sum of weights[i] for each position where vector[i] == 0
    #    Vectors with more zeros in high-weight positions get higher priority.
    zero_score = 0.0
    for i, v in enumerate(vector):
        if v == 0:
            zero_score += weights[i]

    # 2. Balance penalty (light): discourage extreme vectors (all same digit)
    c0 = c1 = c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    target = n / 3.0
    balance_penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2

    # 3. Light autocorrelation penalty (lag 1) to avoid local patterns
    auto_corr = 0
    for i in range(n - 1):
        auto_corr += vector[i] * vector[i + 1]

    # 4. Small deterministic hash for tie-breaking
    h = 0
    for v in vector:
        h = (h * 31 + v) & 0xFFFFFFFF
    hash_term = h / 4294967295.0

    # Combine: zero_score is in [0, sum(weights)] ~ [0, n]
    # balance_penalty is in [0, n^2/3] ~ [0, ~50] for n=12
    # auto_corr is in [0, 4(n-1)] ~ [0, 44] for n=12
    # We want zero_score to dominate, with balance and auto as secondary.

    score = 5.0 * zero_score - 0.5 * balance_penalty - 0.3 * auto_corr + 0.01 * hash_term

    return score
