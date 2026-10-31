# EVOLVE-BLOCK-START
import time

_start_time = None
_deadline = 320.0  # safety margin before the 330s kill

def priority(vector, n):
    """
    Priority for greedy cap-set construction in {0,1,2}^n.

    Structural Strategy (Refined with Multiplicative Complexity):
    1. Quadratic Distinctness: Strongly prefer vectors using all 3 symbols.
       Smooth quadratic scaling (0, 1, 4, 9) creates a steep but continuous
       gradient, avoiding the discrete cliffs of linear tiering.
    2. Enhanced Variance Penalty: Strongly penalize high coordinate variance.
       Vectors with many 0s and 2s (high variance) are structurally more likely
       to complete affine lines. Increased weight reflects the multiplicative
       insight that 'extreme' coordinates create rank-deficient patterns.
    3. Middle-Value Bonus: Small bonus for vectors with many 1s. Since 1 is
       the multiplicative identity and the 'neutral' element, vectors centered
       around 1 resist the additive dependencies that form affine lines.
    4. L2 Balance Penalty: Penalize deviation from the uniform symbol distribution
       (n/3, n/3, n/3) using squared error. This is a smooth surrogate for balance.
    5. Symmetry Breaking: Favor vectors with more 2s to prevent clustering
       in the {0,1}^n sub-cube and ensure uniform exploration of the space.
    6. Deterministic Hash: Fast polynomial hash for tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _deadline:
        # Past deadline: ordering no longer matters, return a flat constant.
        return 0.0

    # Fast symbol counting in one pass (collects counts, sum, and sum-of-squares)
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

    # Factor 1: Smooth quadratic distinct-symbol score.
    # Quadratic in presence count: 0 -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Enhanced variance-based danger proxy.
    # High coordinate variance means more "extreme" coordinates (0s and 2s),
    # which are structurally more likely to complete an affine line.
    # Increased weight from 5000 to 7000 to strengthen the multiplicative
    # complexity heuristic: extreme values create rank-deficient patterns.
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 7000.0

    # Factor 3: Middle-value bonus (multiplicative complexity refinement).
    # Vectors with many 1s are "neutral" in both additive and multiplicative
    # senses, resisting the formation of affine lines. Small positive bonus.
    middle_bonus = c1 * 30.0

    # Factor 4: Quadratic balance penalty (smooth surrogate for L1).
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 5: Symmetry breaking via count of 2s.
    # Favors vectors with more 2s to break the symmetry of the ternary grid
    # and prevent the greedy algorithm from clustering in the {0,1}^n sub-cube.
    weight_score = c2 * 100.0

    # Factor 6: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + danger_score + middle_bonus + balance_score + weight_score + hash_score
    return float(total)
# EVOLVE-BLOCK-END
