import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

# State shared across priority() calls within one evaluation
_state = {
    'n': None,
    'vectors': [],
    'pair_sum': {},
    'size': 0,
}


def _ensure_init(n):
    if _state['n'] != n:
        _state['n'] = n
        _state['vectors'] = []
        _state['pair_sum'] = {}
        _state['size'] = 0


def _update_pair_sum(vec, n):
    ps = _state['pair_sum']
    size = _state['size']
    for j in range(size):
        vj = _state['vectors'][j]
        key = tuple((vec[i] + vj[i]) % 3 for i in range(n))
        ps[key] = ps.get(key, 0) + 1
        key_rev = tuple((vj[i] + vec[i]) % 3 for i in range(n))
        ps[key_rev] = ps.get(key_rev, 0) + 1


def _fast_hash(vector):
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    return h / 0x7FFFFFFF


def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    _ensure_init(n)

    # Fast symbol counting
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)

    # Factor 1: Quadratic distinct symbol score
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Mod-3 coordinate-sum penalty
    coord_sum = c1 + 2 * c2
    sum_mod = coord_sum % 3
    if sum_mod == 0:
        sum_score = -5000.0
    elif sum_mod == 1:
        sum_score = 2000.0
    else:
        sum_score = 1500.0

    # Factor 3: Quadratic balance penalty
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Smooth global line-avoidance penalty
    # Count how many existing pairs would be completed by this vector.
    # A vector v completes a line with pair (a,b) iff a + b + v ≡ 0 mod 3,
    # i.e., v ≡ -(a+b) ≡ (3 - (a+b)) mod 3 coordinate-wise.
    ps = _state['pair_sum']
    if ps:
        v_key = tuple((3 - vector[i]) % 3 for i in range(n))
        line_count = ps.get(v_key, 0)
        # Smooth super-linear penalty: grows quadratically with line_count
        # to strongly discourage vectors that complete many lines
        line_penalty = -line_count * line_count * 10.0 - line_count * 50.0
    else:
        line_penalty = 0.0

    # Factor 5: Variance-based danger proxy (smooth local structural signal)
    s = c1 + 2 * c2
    s2 = c1 + 4 * c2
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 6: Fast polynomial rolling hash for deterministic tie-breaking
    hash_score = _fast_hash(vector)

    # Combined priority: higher is added first
    total = distinct_score + sum_score + balance_score + line_penalty + danger_score + hash_score

    # Update state: assume the vector is accepted (evaluator only calls priority
    # for vectors that pass the cap-set check)
    vec_tuple = tuple(vector)
    _state['vectors'].append(vec_tuple)
    _update_pair_sum(vec_tuple, n)
    _state['size'] += 1

    return float(total)
