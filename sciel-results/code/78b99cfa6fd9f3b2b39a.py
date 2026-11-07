"""Maximum sum of pairwise-coprime integers in 1..256 via greedy + local search."""
import time
import math
from math import gcd

# EVOLVE-BLOCK-START
N = 256

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _prime_factors(x):
    pf = set()
    d = 2
    t = x
    while d * d <= t:
        if t % d == 0:
            pf.add(d)
            while t % d == 0:
                t //= d
        d += 1
    if t > 1:
        pf.add(t)
    return pf

# Precompute prime factors for 2..256
PF = [set()] * (N + 1)
for x in range(2, N + 1):
    PF[x] = _prime_factors(x)

def _conflict(a, b):
    if a == 1 or b == 1:
        return False
    return bool(PF[a] & PF[b])

def _greedy_by_weight(order):
    # order: list of integers to consider
    chosen = []
    for x in order:
        if x == 1:
            continue
        ok = True
        for y in chosen:
            if _conflict(x, y):
                ok = False
                break
        if ok:
            chosen.append(x)
    return chosen

def _local_search(initial):
    s = set(initial)
    cur = sum(s)
    improved = True
    guard = 0
    while improved and guard < 200:
        improved = False
        guard += 1
        # 1-for-1 swaps
        for v in list(s):
            s.remove(v)
            best_add = -1
            best_gain = 0
            for u in range(2, N + 1):
                if u in s:
                    continue
                if u == v:
                    continue
                ok = True
                for x in s:
                    if _conflict(u, x):
                        ok = False
                        break
                if ok:
                    gain = u - v
                    if gain > best_gain:
                        best_gain = gain
                        best_add = u
            if best_add >= 0:
                s.add(best_add)
                cur += best_gain
                improved = True
            else:
                s.add(v)
        # 1-for-2 swaps
        for v in list(s):
            s.remove(v)
            candidates = []
            for u in range(2, N + 1):
                if u in s:
                    continue
                ok = True
                for x in s:
                    if _conflict(u, x):
                        ok = False
                        break
                if ok:
                    candidates.append(u)
            best_pair_gain = 0
            best_pair = None
            cl = len(candidates)
            for i in range(cl):
                for j in range(i + 1, cl):
                    u, w = candidates[i], candidates[j]
                    if _conflict(u, w):
                        continue
                    gain = u + w - v
                    if gain > best_pair_gain:
                        best_pair_gain = gain
                        best_pair = (u, w)
            if best_pair:
                s.add(best_pair[0])
                s.add(best_pair[1])
                cur += best_pair_gain
                improved = True
            else:
                s.add(v)
        # add single
        for u in range(2, N + 1):
            if u in s:
                continue
            ok = True
            for x in s:
                if _conflict(u, x):
                    ok = False
                    break
            if ok:
                s.add(u)
                cur += u
                improved = True
    return cur, s

def construct():
    start = time.time()
    deadline = start + 75.0
    # Try multiple greedy orders
    best_set = set()
    best_val = 0
    orders = [
        list(range(2, N + 1, -1)),  # descending
        list(range(2, N + 1)),  # ascending
        sorted(range(2, N + 1), key=lambda x: -len(PF[x])),  # fewer factors first
        sorted(range(2, N + 1), key=lambda x: (x, -len(PF[x]))),
    ]
    for order in orders:
        if time.time() > deadline:
            break
        chosen = _greedy_by_weight(order)
        val, s = _local_search(chosen)
        if val > best_val:
            best_val = val
            best_set = s
    # Final result
    result = [1] + list(best_set)
    result = list(set(result))
    result = [x for x in result if 1 <= x <= 256]
    result.sort()
    # Verify
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                result = [1] + [x for x in range(2, N + 1) if _is_prime(x)]
                break
    return result
# EVOLVE-BLOCK-END
