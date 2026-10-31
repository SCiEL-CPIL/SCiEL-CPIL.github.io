import time

_start_time = None
_DEADLINE = 310.0  # safety margin below 330s


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (Refined Combined Design):
    1. Primary: Strong quadratic distinct-symbol score favoring vectors with all 3 symbols.
    2. Secondary: Variance penalty - high variance (many 0s/2s) indicates line-completion risk.
    3. Tertiary: Quadratic balance penalty for deviation from uniform (n/3 each) distribution.
    4. Quaternary: Mild symmetry breaking favoring more 2s to avoid {0,1}^n clustering.
    5. Quinary: Fast rolling hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Single-pass counting
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
        else:
            c2 += 1
            s += 2
            s2 += 4

    # Factor 1: Quadratic distinct-symbol score (dominant term)
    # 0 symbols -> 0, 1 -> 25000, 2 -> 100000, 3 -> 225000
    nd = (1 if c0 else 0) + (1 if c1 else 0) + (1 if c2 else 0)
    distinct_score = 25000.0 * nd * nd

    # Factor 2: Variance-based danger penalty
    # High variance = many extreme coords (0,2) = more line risk
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 8000.0

    # Factor 3: Quadratic balance penalty
    # Deviation from uniform (n/3, n/3, n/3)
    ideal = n / 3.0
    dev_sq = (c0 - ideal) * (c0 - ideal) + (c1 - ideal) * (c1 - ideal) + (c2 - ideal) * (c2 - ideal)
    balance_score = -dev_sq * 10.0

    # Factor 4: Symmetry breaking (mild)
    weight_score = c2 * 50.0

    # Factor 5: Fast rolling hash for tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    total = distinct_score + weight_score + balance_score + danger_score + hash_score
    return float(total)
