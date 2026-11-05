import time

# EVOLVE-BLOCK-START

def _gen_smooth(limit):
    """Generate all 2,3,5-smooth numbers up to limit."""
    vals = set()
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                vals.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(v for v in vals if v <= limit)

# Precompute smooth numbers up to 128
_SMOOTH = _gen_smooth(128)
_N = len(_SMOOTH)
_SMOOTH_SET = set(_SMOOTH)

# Precompute divisibility incompatibility masks
_INCOMPAT_MASK = [0] * _N
for i in range(_N):
    vi = _SMOOTH[i]
    mask = 1 << i
    for j in range(_N):
        if i == j:
            continue
        vj = _SMOOTH[j]
        if vi > vj:
            if vi % vj == 0:
                mask |= (1 << j)
        else:
            if vj % vi == 0:
                mask |= (1 << j)
    _INCOMPAT_MASK[i] = mask

# Precompute suffix sums of the sorted smooth list for O(1) pruning.
# _SUF[i] = sum(_SMOOTH[i:]) for i in 0.._N; _SUF[_N] = 0.
_SUF = [0] * (_N + 1)
for i in range(_N - 1, -1, -1):
    _SUF[i] = _SUF[i + 1] + _SMOOTH[i]

def _popcount(x):
    return x.bit_count()

def _next_avail(mask, start):
    """Return smallest index >= start with a bit set in mask, or _N if none."""
    m = mask >> start
    if m == 0:
        return _N
    return start + (m & (-m)).bit_length() - 1

def _find_minimal_rep(target, k):
    """
    Find k distinct smooth numbers summing to target, pairwise non-dividing.
    Returns a list of values or None.
    """
    if k == 1:
        if target in _SMOOTH_SET:
            return [target]
        return None

    full_mask = (1 << _N) - 1
    path = [0] * k

    def dfs(remaining_k, remaining_sum, available_mask, depth):
        if remaining_k == 0:
            return remaining_sum == 0
        if remaining_sum <= 0:
            return False

        # Feasibility: enough available elements?
        if _popcount(available_mask) < remaining_k:
            return False

        # Prune via precomputed suffix sums over the sorted smooth list.
        # Smallest available index:
        lo = _next_avail(available_mask, 0)
        if lo == _N:
            return False
        min_sum = _SUF[lo] - _SUF[lo + remaining_k]
        if remaining_sum < min_sum:
            return False
        # Largest available index:
        hi = _N - 1 - _next_avail(available_mask >> (_N - 1), 0)
        max_sum = _SUF[hi - remaining_k + 1] - _SUF[hi + 1]
        if remaining_sum > max_sum:
            return False

        # Iterate candidates from largest to smallest.
        for i in range(_N - 1, -1, -1):
            if not (available_mask >> i) & 1:
                continue
            v = _SMOOTH[i]
            if v > remaining_sum:
                continue

            path[depth] = i
            new_mask = available_mask & ~_INCOMPAT_MASK[i]

            if dfs(remaining_k - 1, remaining_sum - v, new_mask, depth + 1):
                return True

        return False

    if dfs(k, target, full_mask, 0):
        return [_SMOOTH[i] for i in path[:k]]
    return None

def _quick_fallback(n):
    if n in _SMOOTH_SET:
        return [n]
    for a in _SMOOTH:
        if a >= n:
            break
        b = n - a
        if b in _SMOOTH_SET and a != b and a % b != 0 and b % a != 0:
            return [a, b]
    for a in _SMOOTH:
        if a >= n:
            break
        for b in _SMOOTH:
            if b <= a:
                continue
            if a + b >= n:
                break
            c = n - a - b
            if c in _SMOOTH_SET and c > b:
                if (a % b != 0 and b % a != 0 and a % c != 0 and c % a != 0
                        and b % c != 0 and c % b != 0):
                    return [a, b, c]
    return [1] if n == 1 else [n]

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    results = [None] * 129

    for n in range(1, 129):
        if time.time() - start_time > DEADLINE:
            results[n] = _quick_fallback(n)
            continue

        found = None
        for k in range(1, 31):
            if k > n:
                break
            if n < k * (k + 1) // 2:
                break
            found = _find_minimal_rep(n, k)
            if found is not None:
                break

        if found is None:
            found = _quick_fallback(n)

        results[n] = found

    return results[1:129]

# EVOLVE-BLOCK-END
