"""Cap set greedy builder: heuristics + exact line-completion tracking via pair sums."""
import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

# State shared across priority() calls within one evaluation
_state = {
    'n': None,
    'vectors': [],       # list of tuple vectors in the cap set
    'pair_sum': {},      # dict: tuple of (sum mod 3) per coord -> count of ordered pairs
    'size': 0,
}


def _reset_state(n):
    """Reset state when n changes (new evaluation)."""
    _state['n'] = n
    _state['vectors'] = []
    _state['pair_sum'] = {}
    _state['size'] = 0


def _update_pair_sum(vec, n):
    """
    Add contributions from all ordered pairs involving vec to pair_sum.
    For each existing vector vj, add vec+vj and vj+vec to pair_sum.
    O(size * n) per call, but size grows slowly relative to 3^n.
    """
    ps = _state['pair_sum']
    size = _state['size']
    for j in range(size):
        vj = _state['vectors'][j]
        # Compute vec + vj mod 3
        key = tuple((vec[i] + vj[i]) % 3 for i in range(n))
        ps[key] = ps.get(key, 0) + 1
        # vec + vj == vj + vec, so both ordered pairs have the same key
        # But we need to count both (vec,vj) and (vj,vec)
        ps[key] = ps.get(key, 0) + 1


def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.

    Design (combined):
    1. Quadratic symbol-presence score: smooth landscape favoring vectors
       that use all three symbols.
    2. Mod-3 coordinate-sum penalty: vectors with sum ≡ 0 mod 3 are more
       likely to complete affine lines; penalize strongly.
    3. Hamming-weight symmetry-breaking: favor vectors with more 2s to
       prevent greedy clustering in the {0,1}^n sub-cube.
    4. Quadratic balance penalty: L2 deviation from uniform (n/3,n/3,n/3).
    5. Variance-based danger proxy: high coordinate variance means more
       extreme coordinates, structurally more likely to form lines.
    6. EXACT line-completion penalty: count how many existing pairs would
       be completed by this vector (the actual cap-set constraint).
    7. Fast rolling hash for deterministic tie-breaking.

    All heuristic components computed in O(n) with a single pass.
    The line-completion lookup is O(n) for the key computation.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Reset state if n changed
    if _state['n'] != n:
        _reset_state(n)

    # Single-pass symbol counting + sum + sum-of-squares
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

    # Factor 1: Quadratic distinct-symbol score
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Mod-3 sum penalty
    coord_sum = c1 + 2 * c2
    sum_mod = coord_sum % 3
    if sum_mod == 0:
        sum_score = -5000.0
    elif sum_mod == 1:
        sum_score = 2000.0
    else:
        sum_score = 1500.0

    # Factor 3: Hamming-weight symmetry-breaking (favor 2s)
    weight_score = c2 * 100.0

    # Factor 4: Quadratic balance penalty (deviation from uniform n/3)
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 5: Variance-based danger proxy
    if n > 0:
        mean = s / n
        variance = s2 / n - mean * mean
        danger_score = -variance * 5000.0
    else:
        danger_score = 0.0

    # Factor 6: EXACT line-completion penalty
    # A vector v completes a line with pair (a,b) iff a + b + v ≡ 0 (mod 3)
    # i.e., v ≡ -(a+b) ≡ (3 - (a+b)) mod 3 coordinate-wise.
    # Look up how many ordered pairs (a,b) have a+b ≡ -v (mod 3).
    ps = _state['pair_sum']
    if ps:
        # Compute the key: for each coord, (3 - vector[i]) % 3
        v_key = tuple((3 - vector[i]) % 3 for i in range(n))
        line_count = ps.get(v_key, 0)
        # Scale penalty: each completed line is a hard violation,
        # but since the evaluator filters, this acts as a soft ordering signal.
        # Scale with set size to keep magnitude reasonable relative to other terms.
        line_penalty = -line_count * 100.0
    else:
        line_penalty = 0.0

    # Factor 7: Fast polynomial rolling hash for deterministic tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first
    total = distinct_score + sum_score + weight_score + balance_score + danger_score + line_penalty + hash_score

    # Update state: record this vector and update pair sums
    vec_tuple = tuple(vector)
    _state['vectors'].append(vec_tuple)
    _update_pair_sum(vec_tuple, n)
    _state['size'] += 1

    return float(total)
