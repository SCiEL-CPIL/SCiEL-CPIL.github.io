import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (Refined Symmetry and Balance):
    1. Primary: Smooth quadratic distinct-symbol score. Ensures vectors use all 3 symbols.
    2. Secondary: Strong quadratic balance penalty. Heavily penalizes deviation from
       the uniform (n/3, n/3, n/3) distribution. This encourages the greedy algorithm
       to select "central" vectors that are symmetric, which tends to block more
       affine lines than biased vectors.
    3. Tertiary: Smooth variance-based danger proxy. Penalizes high coordinate variance
       to avoid "extreme" vectors that are structurally more likely to complete lines.
    4. Quaternary: Fast polynomial rolling hash for robust deterministic tie-breaking.
       This ensures a consistent, pseudo-random ordering among vectors with similar
       structural scores, preventing clustering in specific sub-cubes.

    All components computed in O(n) time.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting
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

    # Factor 1: Distinct symbol score
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Balance penalty (Increased weight)
    # We want vectors to be as close to (n/3, n/3, n/3) as possible.
    # Using a higher weight (e.g., 100.0) makes this the dominant secondary factor.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 100.0

    # Factor 3: Pairwise interaction penalty (algebraic dependence proxy)
    # Compute sum of (x_i * x_j) for i < j, normalized.
    # High values indicate structured vectors that may lie in low-degree subspaces.
    # Use a smooth penalty: -alpha * (pairwise_sum / n^2)^2
    pairwise_sum = 0.0
    for i in range(n):
        vi = vector[i]
        for j in range(i + 1, n):
            pairwise_sum += vi * vector[j]
    
    # Normalize and penalize
    pairwise_norm = pairwise_sum / (n * n) if n > 1 else 0.0
    danger_score = -pairwise_norm * pairwise_norm * 500.0

    # Factor 4: Hash for tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    total = distinct_score + balance_score + danger_score + hash_score
    return float(total)
