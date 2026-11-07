import time
import random

_MAX = 256
_MAXN = 60

def _check(sums, n):
    for i in range(n + 1):
        si = sums[i]
        if si == 0:
            continue
        for j in range(i + 1, n + 1):
            if si & sums[j]:
                return False
    return True

def _try_add(sums, n, x, maxn):
    if n >= maxn:
        return None
    new = [0] * (n + 2)
    new[0] = sums[0]
    for k in range(1, n + 1):
        new[k] = sums[k] | (sums[k - 1] << x)
    new[n + 1] = sums[n] << x
    if _check(new, n + 1):
        return new
    return None

def _greedy(order, maxn, deadline):
    sums = [0] * (maxn + 1)
    sums[0] = 1
    chosen = []
    n = 0
    for x in order:
        if n >= maxn or time.time() > deadline:
            break
        new = _try_add(sums, n, x, maxn)
        if new is not None:
            sums = new
            chosen.append(x)
            n += 1
    return chosen

def _build_sums(chosen):
    sums = [0] * (len(chosen) + 1)
    sums[0] = 1
    n = 0
    for x in chosen:
        new = [0] * (n + 2)
        new[0] = sums[0]
        for k in range(1, n + 1):
            new[k] = sums[k] | (sums[k - 1] << x)
        new[n + 1] = sums[n] << x
        sums = new
        n += 1
    return sums

def _local_search(chosen, maxn, deadline):
    n = len(chosen)
    sums = _build_sums(chosen)
    chosen_set = set(chosen)
    while n < maxn and time.time() < deadline:
        added = False
        for x in range(_MAX, 0, -1):
            if x in chosen_set:
                continue
            new = _try_add(sums, n, x, maxn)
            if new is not None:
                sums = new
                chosen.append(x)
                chosen_set.add(x)
                n += 1
                added = True
                break
        if added:
            continue
        swapped = False
        for i in range(n):
            if time.time() > deadline:
                break
            ci = chosen[i]
            for x in range(_MAX, 0, -1):
                if x in chosen_set:
                    continue
                cand = chosen[:i] + chosen[i + 1:] + [x]
                s2 = _build_sums(cand)
                if not _check(s2, n):
                    continue
                cs2 = set(cand)
                for y in range(_MAX, 0, -1):
                    if y in cs2:
                        continue
                    new2 = _try_add(s2, n, y, maxn)
                    if new2 is not None:
                        chosen = cand + [y]
                        chosen_set = cs2 | {y}
                        sums = new2
                        n += 1
                        swapped = True
                        break
                if swapped:
                    break
            if swapped:
                break
        if not swapped:
            break
    return chosen

def construct():
    start = time.time()
    deadline = start + 75.0
    rng = random.Random(12345)
    best = []
    for o in (list(range(_MAX, 0, -1)), list(range(1, _MAX + 1))):
        if time.time() > deadline:
            break
        res = _greedy(o, _MAXN, deadline)
        if len(res) > len(best):
            best = res
    while time.time() < deadline - 15.0:
        o = list(range(1, _MAX + 1))
        rng.shuffle(o)
        res = _greedy(o, _MAXN, deadline)
        if len(res) > len(best):
            best = res
    if time.time() < deadline:
        best = _local_search(best, _MAXN, deadline)
    return [int(x) for x in best]
