import time
import random
import math

N = 128

# Precompute all nonconstant 4-term APs (0-indexed positions 0..127)
_aps = []
_pos_aps = [[] for _ in range(N)]
for _d in range(1, N // 3 + 1):
    for _a in range(0, N - 3 * _d):
        _ap = (_a, _a + _d, _a + 2 * _d, _a + 3 * _d)
        _idx = len(_aps)
        _aps.append(_ap)
        for _p in _ap:
            _pos_aps[_p].append(_idx)

_APS = _aps
_POS_APS = _pos_aps
_NUM_APS = len(_APS)


def _init_cost(c):
    ap_bad = [0] * _NUM_APS
    total = 0
    for idx, ap in enumerate(_APS):
        d = len({c[i] for i in ap})
        if d <= 2:
            ap_bad[idx] = 1
            total += 1
    return ap_bad, total


def _sa_run(c, k, budget):
    ap_bad, total = _init_cost(c)
    best_c = list(c)
    best_total = total
    t0 = time.time()
    T = 3.0
    cool = 0.99995
    rnd = random
    exp = math.exp
    while time.time() - t0 < budget:
        p = rnd.randrange(N)
        newc = rnd.randrange(k)
        if newc == c[p]:
            continue
        oldc = c[p]
        old_total = total
        c[p] = newc
        delta = 0
        for idx in _POS_APS[p]:
            ap = _APS[idx]
            d = len({c[i] for i in ap})
            bad = 1 if d <= 2 else 0
            if bad != ap_bad[idx]:
                delta += bad - ap_bad[idx]
                ap_bad[idx] = bad
        new_total = old_total + delta
        if new_total < best_total:
            best_total = new_total
            best_c = list(c)
        if new_total <= old_total or rnd.random() < exp(-delta / T):
            total = new_total
        else:
            c[p] = oldc
            for idx in _POS_APS[p]:
                ap = _APS[idx]
                d = len({c[i] for i in ap})
                bad = 1 if d <= 2 else 0
                ap_bad[idx] = bad
            total = old_total
        T *= cool
    return best_c, best_total


def construct():
    deadline = time.time() + 55.0
    # Guaranteed valid fallback: each position a unique color.
    base = list(range(N))
    best_valid = base
    best_distinct = N

    def _try(k, budget, starts):
        nonlocal best_valid, best_distinct
        t_end = time.time() + budget
        it = 0
        while time.time() < t_end and time.time() < deadline - 2.0:
            start = list(starts[it % len(starts)])
            c, tot = _sa_run(start, k, max(0.3, min(5.0, t_end - time.time())))
            if tot == 0:
                used = sorted(set(c))
                m = len(used)
                if m < best_distinct:
                    remap = {v: i for i, v in enumerate(used)}
                    best_valid = [remap[x] for x in c]
                    best_distinct = m
                if m <= 3:
                    return True
            it += 1
        return False

    # Palette sizes in ascending order; smaller is better.
    plan = [
        (3, 25.0, [[i % 3 for i in range(N)],
                   [i % 3 for i in range(N)] if N % 2 else [i % 3 for i in range(N)],
                   [random.randrange(3) for _ in range(N)]]),
        (4, 15.0, [[i % 4 for i in range(N)],
                   [random.randrange(4) for _ in range(N)]]),
        (5, 8.0, [[i % 5 for i in range(N)],
                  [random.randrange(5) for _ in range(N)]]),
        (6, 5.0, [[i % 6 for i in range(N)],
                  [random.randrange(6) for _ in range(N)]]),
        (8, 4.0, [[i % 8 for i in range(N)],
                  [random.randrange(8) for _ in range(N)]]),
    ]
    for k, budget, starts in plan:
        if time.time() > deadline - 5.0:
            break
        b = min(budget, deadline - time.time())
        if b < 2.0:
            break
        if _try(k, b, starts):
            break
    return best_valid
