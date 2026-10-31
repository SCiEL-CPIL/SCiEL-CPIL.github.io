import time
import hashlib

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

# Precomputed weights for monomial basis projection
# w0: constant term weight
# w1[i]: linear term weights (deterministic pseudo-random)
_w1 = None
_w1_sum_sq = 0.0


def _init_weights(n_max=64):
    """Initialize deterministic weights for the linear basis projection."""
    global _w1, _w1_sum_sq
    if _w1 is not None:
        return
    # Generate deterministic pseudo-random weights using hashlib
    _w1 = []
    for i in range(n_max):
        h = hashlib.md5(f"lin_{i}".encode()).digest()
        # Convert first 8 bytes to a float in [-1, 1]
        val = int.from_bytes(h[:4], 'big') / (2**32) - 0.5
        _w1.append(val)
    _w1_sum_sq = sum(w * w for w in _w1)


_init_weights()


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (combined design, smooth surrogates):
    1. Primary: Smooth quadratic distinct-symbol score. Quadratic in the number
       of present symbols (0,1,4,9 * 10000) gives a smoother landscape than the
       discrete 10000/20000/30000 step, producing a better gradient signal for
       the greedy ordering.
    2. Secondary: Hamming weight relative to all-ones (count of 2s). Favors
       vectors "heavy" in the 2-coordinate to break the symmetry of the ternary
       grid and prevent the greedy algorithm from clustering in the {0,1}^n
       sub-cube, forcing uniform exploration of the full {0,1,2}^n space.
    3. Tertiary: Quadratic balance penalty. L2 deviation from the uniform
       (n/3, n/3, n/3) distribution, a smooth surrogate for the L1 deviation,
       provides a smoother gradient than the piecewise-linear L1 penalty.
    4. Quaternary: Smooth variance-based danger proxy. High coordinate variance
       means more "extreme" coordinates, which are structurally more likely to
       complete an affine line, so we penalize variance.
    5. Quinary: Fast polynomial rolling hash for robust deterministic
       tie-breaking.

    All components computed in O(n) time with minimal per-call overhead.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0

    # Fast symbol counting in one pass (collects counts, sum, and sum-of-squares)
    # Also compute projection onto orthogonal complement of linear basis
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    proj = 0.0  # Projection onto orthogonal complement of span{1, x_1, ..., x_n}
    
    for i, v in enumerate(vector):
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
        # Accumulate weighted sum for projection (linear term)
        if i < len(_w1):
            proj += _w1[i] * v

    # Factor 1: Smooth quadratic distinct-symbol score.
    # Quadratic in presence count: 0 -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000.
    # Smoother than the discrete step, gives a better gradient for the ordering.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Hamming weight relative to all-ones (count of 2s).
    # Favors vectors with more 2s to break symmetry and explore the full space,
    # preventing clustering in the {0,1}^n sub-cube.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty (smooth surrogate for L1).
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Smooth variance-based danger proxy.
    # High coordinate variance = more extreme coordinates = higher line risk.
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 5: Projection magnitude onto orthogonal complement of linear basis.
    # The projection of vector x onto the orthogonal complement of span{1, x_1, ..., x_n}
    # has magnitude squared = ||x||^2 - (projection onto span)^2
    # Projection onto span{1} component: (sum(x_i) / n) * 1_vector
    # Projection onto span{x_i} component: weighted linear combination
    # We approximate the orthogonal complement magnitude by:
    #   ||x||^2 - (s/n)^2 * n - proj^2 / _w1_sum_sq  (simplified)
    # Higher values indicate more "algebraic independence" from low-degree monomials,
    # which structurally prevents affine line formation.
    norm_sq = s2
    const_proj_sq = (s * s) / n  # Squared norm of projection onto constant vector
    lin_proj_sq = (proj * proj) / _w1_sum_sq if _w1_sum_sq > 0 else 0.0
    # Orthogonal complement magnitude squared (approximate)
    ortho_comp_sq = norm_sq - const_proj_sq - lin_proj_sq
    if ortho_comp_sq < 0:
        ortho_comp_sq = 0.0
    # Scale to be comparable with other terms
    projection_score = ortho_comp_sq * 2000.0

    # Factor 6: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + weight_score + balance_score + danger_score + projection_score + hash_score
    return float(total)
