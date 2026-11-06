import time
from itertools import combinations

def _smooth_numbers(limit):
    s = set()
    a = 1
    while a <= limit:
        b = a
        while b <= limit:
            c = b
            while c <= limit:
                s.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(s)

_SMOOTH = _smooth_numbers(128)
_SMOOTH_SET = set(_SMOOTH)
_N = len(_SMOOTH)

# Precompute incompatibility: for each index, set of incompatible indices
_incompat = [set() for _ in range(_N)]
for i in range(_N):
    for j in range(i + 1, _N):
        a, b = _SMOOTH[i], _SMOOTH[j]
        if a % b == 0 or b % a == 0:
            _incompat[i].add(j)
            _incompat[j].add(i)

# Precompute valid antichains for k=2,3,4,5,6 and map by sum
# This ensures we find the minimum k for each target efficiently
# and avoids per-target search overhead.

# k=1: trivial
# k=2: pairs (i, j) with i < j, SMOOTH[i] + SMOOTH[j] <= 128, antichain
_antichain_sums = {k: {} for k in range(1, 7)}  # k -> {sum: [indices]}

# k=1
for i, v in enumerate(_SMOOTH):
    if v <= 128:
        if v not in _antichain_sums[1]:
            _antichain_sums[1][v] = []
        _antichain_sums[1][v].append([i])

# k=2
for i in range(_N):
    for j in range(i + 1, _N):
        s = _SMOOTH[i] + _SMOOTH[j]
        if s <= 128 and i not in _incompat[j]:
            if s not in _antichain_sums[2]:
                _antichain_sums[2][s] = []
            _antichain_sums[2][s].append([i, j])

# k=3
for i in range(_N):
    for j in range(i + 1, _N):
        if j in _incompat[i]:
            continue
        for k in range(j + 1, _N):
            if k in _incompat[i] or k in _incompat[j]:
                continue
            s = _SMOOTH[i] + _SMOOTH[j] + _SMOOTH[k]
            if s <= 128:
                if s not in _antichain_sums[3]:
                    _antichain_sums[3][s] = []
                _antichain_sums[3][s].append([i, j, k])

# k=4
for i in range(_N):
    for j in range(i + 1, _N):
        if j in _incompat[i]:
            continue
        for k in range(j + 1, _N):
            if k in _incompat[i] or k in _incompat[j]:
                continue
            for l in range(k + 1, _N):
                if l in _incompat[i] or l in _incompat[j] or l in _incompat[k]:
                    continue
                s = _SMOOTH[i] + _SMOOTH[j] + _SMOOTH[k] + _SMOOTH[l]
                if s <= 128:
                    if s not in _antichain_sums[4]:
                        _antichain_sums[4][s] = []
                    _antichain_sums[4][s].append([i, j, k, l])

# k=5
for i in range(_N):
    for j in range(i + 1, _N):
        if j in _incompat[i]:
            continue
        for k in range(j + 1, _N):
            if k in _incompat[i] or k in _incompat[j]:
                continue
            for l in range(k + 1, _N):
                if l in _incompat[i] or l in _incompat[j] or l in _incompat[k]:
                    continue
                for m in range(l + 1, _N):
                    if m in _incompat[i] or m in _incompat[j] or m in _incompat[k] or m in _incompat[l]:
                        continue
                    s = _SMOOTH[i] + _SMOOTH[j] + _SMOOTH[k] + _SMOOTH[l] + _SMOOTH[m]
                    if s <= 128:
                        if s not in _antichain_sums[5]:
                            _antichain_sums[5][s] = []
                        _antichain_sums[5][s].append([i, j, k, l, m])

# k=6
for i in range(_N):
    for j in range(i + 1, _N):
        if j in _incompat[i]:
            continue
        for k in range(j + 1, _N):
            if k in _incompat[i] or k in _incompat[j]:
                continue
            for l in range(k + 1, _N):
                if l in _incompat[i] or l in _incompat[j] or l in _incompat[k]:
                    continue
                for m in range(l + 1, _N):
                    if m in _incompat[i] or m in _incompat[j] or m in _incompat[k] or m in _incompat[l]:
                        continue
                    for n in range(m + 1, _N):
                        if (n in _incompat[i] or n in _incompat[j] or n in _incompat[k] or 
                            n in _incompat[l] or n in _incompat[m]):
                            continue
                        s = _SMOOTH[i] + _SMOOTH[j] + _SMOOTH[k] + _SMOOTH[l] + _SMOOTH[m] + _SMOOTH[n]
                        if s <= 128:
                            if s not in _antichain_sums[6]:
                                _antichain_sums[6][s] = []
                            _antichain_sums[6][s].append([i, j, k, l, m, n])

def _find_min_antichain(target, deadline=None):
    """Find minimum number of distinct smooth numbers forming antichain summing to target."""
    if deadline is not None and time.time() > deadline:
        return None
    
    # Try k=1 through k=6 using precomputed tables
    for k in range(1, 7):
        if target in _antichain_sums[k]:
            idxs = _antichain_sums[k][target][0]
            return [_SMOOTH[i] for i in idxs]
    
    # Fallback: backtracking for k=7..10 if needed
    if deadline is not None and time.time() > deadline:
        return None
    
    for k in range(7, 11):
        found = [None]
        
        def backtrack(start, remaining, count, chosen):
            if deadline is not None and time.time() > deadline:
                return
            if count == k:
                if remaining == 0:
                    found[0] = list(chosen)
                    return
                return
            if remaining <= 0:
                return
            # Prune: need (k-count) numbers each >= _SMOOTH[start]
            if start >= _N:
                return
            min_possible = _SMOOTH[start] * (k - count)
            if min_possible > remaining:
                return
            for idx in range(start, _N):
                if deadline is not None and time.time() > deadline:
                    return
                v = _SMOOTH[idx]
                if v > remaining:
                    break
                compat = True
                for ci in chosen:
                    if idx in _incompat[ci]:
                        compat = False
                        break
                if not compat:
                    continue
                chosen.append(idx)
                backtrack(idx + 1, remaining - v, count + 1, chosen)
                chosen.pop()
                if found[0] is not None:
                    return
        
        backtrack(0, target, 0, [])
        if found[0] is not None:
            return [_SMOOTH[i] for i in found[0]]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    results = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Emergency fallback
            for m in range(n, 129):
                ans = _find_min_antichain(m, deadline)
                if ans is None:
                    if m in _SMOOTH_SET:
                        ans = [m]
                    else:
                        ans = [1, m - 1] if (m - 1) in _SMOOTH_SET else [m]
                results.append(ans)
            return results
        
        ans = _find_min_antichain(n, deadline)
        if ans is None:
            # Last resort fallback
            if n in _SMOOTH_SET:
                ans = [n]
            else:
                # Try to find any valid representation
                found = False
                for k in range(2, 15):
                    smooth_leq = [v for v in _SMOOTH if v <= n]
                    if len(smooth_leq) >= k:
                        for terms in combinations(smooth_leq, k):
                            if sum(terms) == n:
                                valid = True
                                for p in range(len(terms)):
                                    for q in range(p + 1, len(terms)):
                                        if terms[p] % terms[q] == 0 or terms[q] % terms[p] == 0:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if valid:
                                    ans = list(terms)
                                    found = True
                                    break
                    if found:
                        break
                if not found:
                    ans = [n] if n in _SMOOTH_SET else [1, n - 1]
        results.append(ans)
    
    return results
