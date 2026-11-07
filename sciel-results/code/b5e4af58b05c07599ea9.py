import math
import time

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}


def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _squarefree_kernel(n):
    """Return the squarefree part of n (product of primes with odd exponent)."""
    k = n
    d = 2
    kernel = 1
    while d * d <= k:
        if k % d == 0:
            exp = 0
            while k % d == 0:
                k //= d
                exp += 1
            if exp % 2 == 1:
                kernel *= d
        d += 1 if d == 2 else 2
    if k > 1:
        kernel *= k
    return kernel


def _try_complete(target_sf, start, end, max_size, deadline):
    """Find distinct ints in [start, end] (each > P['n']) whose product has
    squarefree kernel == target_sf. Return (max_val, list) or (None, None)."""
    n = P['n']
    if start <= n:
        start = n + 1
    if end > P['maximum_window']:
        end = P['maximum_window']
    if start > end or max_size < 1:
        return None, None

    # Single-element: n_i such that sf(n_i) == target_sf  ->  n_i = target_sf * s^2
    if target_sf * 4 <= end:  # s=1 gives target_sf (<=n, invalid); start s=2
        s = 2
        while True:
            cand = target_sf * s * s
            if cand > end:
                break
            if cand > n:
                return cand, [cand]
            s += 1

    # Two-element: sf(a)*sf(b) == target_sf (mod squares).
    # Enumerate pairs from the low end (best for minimizing max).
    lo = start
    hi = min(end, lo + 4000)
    if hi > end:
        hi = end
    best = None
    a = lo
    while a <= hi:
        if time.time() > deadline:
            break
        kfa = _squarefree_kernel(a)
        # need sf(b) == target_sf * kfa  (mod squares)
        need = target_sf * kfa
        # b = need * s^2
        s = 1
        while True:
            b = need * s * s
            if b > hi or b <= n:
                if b > hi:
                    break
                s += 1
                continue
            if b == a:
                s += 1
                continue
            # distinct and both > n, in window
            mx = max(a, b)
            if best is None or mx < best[0]:
                best = (mx, [a, b])
            s += 1
        a += 1
    if best is not None:
        return best
    return None, None


def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill

    n = P['n']
    target_sf = n  # 210

    # Baseline: single element 210 * 4 = 840
    baseline_val = 210 * 4
    best_val = baseline_val
    best_list = [baseline_val]

    # Try to improve with two elements (and re-check single) in a low window.
    for window_end in (1000, 2000, 3000, 5000, 10210):
        if time.time() > deadline:
            break
        mx, lst = _try_complete(target_sf, n + 1, window_end, P['maximum_size'], deadline)
        if lst is not None:
            # verify exact condition
            prod = n
            for x in lst:
                prod *= x
            if _is_square(prod) and len(set(lst)) == len(lst) and all(
                isinstance(x, int) and not isinstance(x, bool) for x in lst
            ) and all(n < x <= P['maximum_window'] for x in lst) and 1 <= len(lst) <= P['maximum_size']:
                v = max(lst) - n
                if v < best_val:
                    best_val = v
                    best_list = list(lst)

    # Final validation of best_list
    prod = n
    for x in best_list:
        prod *= x
    if not _is_square(prod):
        # fall back to guaranteed valid
        best_list = [210 * 4]

    # Ensure types are exact Python ints in a list
    return [int(x) for x in best_list]
