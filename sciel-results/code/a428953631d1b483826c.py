import time
import random

_start_time = None
_DEADLINE = 300 * 0.95  # 285s safety margin
_random_vectors = {}

def _get_random_vector(n):
    """Get or create a fixed random vector in {-1, 1}^n for chaotic ordering."""
    if n not in _random_vectors:
        random.seed(n * 7919 + 42)  # Deterministic per n
        _random_vectors[n] = tuple(random.choice([-1, 1]) for _ in range(n))
    return _random_vectors[n]

def _fast_hash(vector, n):
    """Fast deterministic hash for tie-breaking."""
    h = 0
    for i in range(n):
        h = (h * 31 + vector[i] + i) & 0x7FFFFFFF
    return h

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Random linear projection. Projects the vector onto a fixed
       random vector in {-1,1}^n, creating a chaotic non-local ordering that
       decouples selection from lattice symmetries and local density traps.
    2. Secondary: Smooth quadratic distinct-symbol score. Strongly prefers
       vectors using all three symbols {0,1,2}.
    3. Tertiary: Quadratic balance penalty. L2 deviation from uniform (n/3, n/3, n/3).
    4. Quaternary: Smooth variance-based danger proxy. Penalizes high variance
       (many extreme 0s and 2s) that are prone to completing affine lines.
    5. Quinary: Fast polynomial rolling hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting in one pass
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

    # Factor 1: Random linear projection - chaotic non-local ordering
    rand_vec = _get_random_vector(n)
    proj = 0
    for i in range(n):
        proj += rand_vec[i] * vector[i]
    # Scale projection to be comparable to other terms; range is [-n, n]
    random_score = proj * 1.0

    # Factor 2: Smooth quadratic distinct-symbol score
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 3: Quadratic balance penalty
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Smooth variance-based danger proxy
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 5: Fast polynomial rolling hash for tie-breaking
    h = _fast_hash(vector, n)
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first
    total = distinct_score + random_score + balance_score + danger_score + hash_score
    return float(total)
