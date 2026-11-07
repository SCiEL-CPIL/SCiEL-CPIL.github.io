import math
import time
from itertools import combinations

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}


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


def _parity_set(n):
    """Return the set of primes with odd exponent in n's factorization.
    Used for XOR-based matching."""
    k = n
    d = 2
    pset = set()
    while d * d <= k:
        if k % d == 0:
            exp = 0
            while k % d == 0:
                k //= d
                exp += 1
            if exp % 2 == 1:
                pset.add(d)
        d += 1 if d == 2 else 2
    if k > 1:
        pset.add(k)
    return frozenset(pset)


def _xor_pset(p1, p2):
    """Symmetric difference of two frozensets."""
    return frozenset(p1.symmetric_difference(p2))


def _validate(lst, n=210, max_window=10210, max_size=100):
    """Check all constraints."""
    if not isinstance(lst, (list, tuple)):
        return False
    if len(lst) < 1 or len(lst) > max_size:
        return False
    if len(set(lst)) != len(lst):
        return False
    for x in lst:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= n or x > max_window:
            return False
    prod = n
    for x in lst:
        prod *= x
    return _is_square(prod)


def _find_best_pair(start, end, target_pset, deadline):
    """Find a pair (a, b) in [start, end] with a < b, both > n,
    such that parity_set(a) XOR parity_set(b) == target_pset.
    Minimize max(a, b)."""
    n = P['n']
    if start <= n:
        start = n + 1
    if end > P['maximum_window']:
        end = P['maximum_window']
    if start >= end:
        return None, None

    # Precompute parity sets for all numbers in range
    range_size = end - start + 1
    if range_size > 50000:
        end = start + 49999

    psets = {}
    for v in range(start, end + 1):
        if time.time() > deadline:
            break
        psets[v] = _parity_set(v)

    # Build a map from parity set to list of values (sorted)
    pset_to_vals = {}
    for v, ps in psets.items():
        if ps not in pset_to_vals:
            pset_to_vals[ps] = []
        pset_to_vals[ps].append(v)

    # For each value a, find b such that pset(b) = pset(a) XOR target_pset
    best = None
    for a in sorted(psets.keys()):
        if time.time() > deadline:
            break
        pa = psets[a]
        needed = _xor_pset(pa, target_pset)
        if needed in pset_to_vals:
            for b in pset_to_vals[needed]:
                if b == a:
                    continue
                mx = max(a, b)
                if best is None or mx < best[0]:
                    best = (mx, [a, b])
                # Since we want min max, and b is in sorted order,
                # the first valid b >= a gives smallest max for this a
                # But we need b != a, so check next if b == a
                if b > a:
                    break
                # b < a, check if there's a larger b
                # Actually we just need any b != a with minimal max
                # Let's continue to find the best b for this a
        # Optimization: if a alone is already larger than current best max, stop
        if best is not None and a >= best[0]:
            break

    if best is not None:
        return best
    return None, None


def _find_best_triple(start, end, target_pset, deadline):
    """Find a triple (a, b, c) in [start, end] with distinct values,
    such that parity_set(a) XOR parity_set(b) XOR parity_set(c) == target_pset.
    Minimize max(a, b, c)."""
    n = P['n']
    if start <= n:
        start = n + 1
    if end > P['maximum_window']:
        end = P['maximum_window']
    if start >= end:
        return None, None

    # Limit search space
    max_range = 3000
    end_eff = min(end, start + max_range)

    psets = {}
    for v in range(start, end_eff + 1):
        if time.time() > deadline:
            break
        psets[v] = _parity_set(v)

    # Build map from parity set to list of values
    pset_to_vals = {}
    for v, ps in psets.items():
        if ps not in pset_to_vals:
            pset_to_vals[ps] = []
        pset_to_vals[ps].append(v)

    vals = sorted(psets.keys())
    best = None

    # For pairs (a, b), check if there exists c with pset(c) = pset(a)^pset(b)^target
    for i, a in enumerate(vals):
        if time.time() > deadline:
            break
        pa = psets[a]
        if best is not None and a >= best[0]:
            break
        for j in range(i + 1, len(vals)):
            b = vals[j]
            if best is not None and b >= best[0]:
                break
            pb = psets[b]
            needed = _xor_pset(_xor_pset(pa, pb), target_pset)
            if needed in pset_to_vals:
                for c in pset_to_vals[needed]:
                    if c == a or c == b:
                        continue
                    mx = max(a, b, c)
                    if best is None or mx < best[0]:
                        best = (mx, [a, b, c])
                    # Found a valid c; since we want min max and c is from list,
                    # take the smallest valid c
                    if c > b:
                        break

    if best is not None:
        return best
    return None, None


def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill

    n = P['n']
    target_pset = _parity_set(n)  # {2, 3, 5, 7}

    # Baseline: single element 210 * 4 = 840
    baseline_val = 210 * 4
    best_val = baseline_val
    best_list = [baseline_val]

    # Try single elements: find smallest x > n such that sf(x) == sf(n)
    # x = 210 * s^2, s >= 2
    for s in range(2, 100):
        cand = n * s * s
        if cand > P['maximum_window']:
            break
        if cand > n:
            v = cand - n
            if v < best_val:
                best_val = v
                best_list = [cand]
            break  # smallest s gives smallest cand

    # Try pairs in increasing windows
    for window_end in (400, 500, 600, 800, 1000, 1500, 2000, 3000, 5000, 10210):
        if time.time() > deadline:
            break
        mx, lst = _find_best_pair(n + 1, window_end, target_pset, deadline)
        if lst is not None and _validate(lst):
            v = max(lst) - n
            if v < best_val:
                best_val = v
                best_list = list(lst)
                # If we found a very good pair, we can stop searching pairs
                if v < 100:
                    break

    # Try triples in a limited window
    if time.time() < deadline - 5:
        for window_end in (400, 500, 600, 800):
            if time.time() > deadline:
                break
            mx, lst = _find_best_triple(n + 1, window_end, target_pset, deadline)
            if lst is not None and _validate(lst):
                v = max(lst) - n
                if v < best_val:
                    best_val = v
                    best_list = list(lst)
                    if v < 50:
                        break

    # Final validation
    if not _validate(best_list):
        best_list = [n * 4]

    return [int(x) for x in best_list]
