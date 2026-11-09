"""Rewritten: greedy prime-power-first + randomized reordering search with deadline guard."""
import math
import time
import random

_START = time.time()
_LIMIT = 55.0  # safety margin under 60s


def _pf(n):
    fs = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            fs.add(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        fs.add(n)
    return fs


_PF = {m: _pf(m) for m in range(2, 129)}


def _build(order):
    chosen = []
    for m in order:
        for r in range(m):
            ok = True
            for r2, m2 in chosen:
                g = math.gcd(m, m2)
                if g > 1 and (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                chosen.append((r, m))
                break
    return chosen


def construct():
    mods = list(range(2, 129))

    def key(m):
        k = len(_PF[m])
        return (k, 0 if k == 1 else 1, m)

    order = sorted(mods, key=key)
    best = _build(order)
    best_len = len(best)

    rng = random.Random(20240517)
    groups = {}
    for m in mods:
        groups.setdefault(len(_PF[m]), []).append(m)
    keys = sorted(groups)

    while time.time() - _START < _LIMIT:
        alt = []
        for k in keys:
            g = groups[k][:]
            rng.shuffle(g)
            alt.extend(g)
        cand = _build(alt)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)

    return [[r, m] for r, m in best]
