"""Fresh design: multi-start greedy (asc + desc) + simulated-annealing local
search over a coverage-count representation, with residue / add / remove moves."""
import time
import random
import math

_BASE = 51975
_FACTORS = [3, 3, 3, 5, 5, 7, 11]


def _all_divisors(n):
    divs = [1]
    for p in _FACTORS:
        cur = []
        for d in divs:
            cur.append(d)
            if d % p != 0:
                cur.append(d * p)
        divs = cur
    return sorted(set(divs))


_MODS = [d for d in _all_divisors(_BASE) if d > 1]
_RESSETS = {}
for _m in _MODS:
    _RESSETS[_m] = [list(range(_r, _BASE, _m)) for _r in range(_m)]


def _score_cnt(cnt):
    s = 0
    for v in cnt:
        if v == 0:
            s += 1
    return s


def _apply(cnt, indices, delta):
    for i in indices:
        cnt[i] += delta


def _rebuild_cnt(classes):
    cnt = [0] * _BASE
    for m, r in classes:
        _apply(cnt, _RESSETS[m][r], 1)
    return cnt


def _greedy_build(order):
    cnt = [0] * _BASE
    classes = []
    used = set()
    for m in order:
        if m in used:
            continue
        ressets = _RESSETS[m]
        best_r, best_gain = 0, -1
        for r in range(m):
            gain = 0
            for i in ressets[r]:
                if cnt[i] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain, best_r = gain, r
        classes.append([m, best_r])
        used.add(m)
        _apply(cnt, ressets[best_r], 1)
    return classes, cnt, _score_cnt(cnt)


def _local_search(classes, cnt, f, deadline, rng):
    used = set(m for m, r in classes)
    best_classes = [c[:] for c in classes]
    best_f = f

    T, T_min = 3.0, 0.01
    n_iters = 15000
    cool = (T_min / T) ** (1.0 / max(1, n_iters))

    for _ in range(n_iters):
        if time.time() > deadline:
            break
        move = rng.random()
        new_classes = new_cnt = None
        new_f = None

        if move < 0.7 and classes:
            ci = rng.randrange(len(classes))
            m, r = classes[ci]
            ressets = _RESSETS[m]
            _apply(cnt, ressets[r], -1)
            unc_total = 0
            for v in cnt:
                if v == 0:
                    unc_total += 1
            best_r, best_f_c = r, f
            for r2 in range(m):
                gain = 0
                for i in ressets[r2]:
                    if cnt[i] == 0:
                        gain += 1
                f_new = unc_total - gain
                if f_new < best_f_c:
                    best_f_c, best_r = f_new, r2
            _apply(cnt, ressets[r], 1)
            if best_r != r:
                new_classes = [c[:] for c in classes]
                new_classes[ci] = [m, best_r]
                new_cnt = cnt[:]
                new_f = best_f_c
        elif move < 0.85 and len(used) < len(_MODS):
            cands = [m for m in _MODS if m not in used]
            if cands:
                m = rng.choice(cands)
                ressets = _RESSETS[m]
                best_r, best_f_c = 0, f
                for r2 in range(m):
                    gain = 0
                    for i in ressets[r2]:
                        if cnt[i] == 0:
                            gain += 1
                    f_new = f - gain
                    if f_new < best_f_c:
                        best_f_c, best_r = f_new, r2
                new_classes = [c[:] for c in classes] + [[m, best_r]]
                new_cnt = cnt[:]
                new_f = best_f_c
        else:
            if classes:
                ci = rng.randrange(len(classes))
                m, r = classes[ci]
                idxs = _RESSETS[m][r]
                unique = 0
                for i in idxs:
                    if cnt[i] == 1:
                        unique += 1
                if unique == 0:
                    new_classes = [c[:] for c in classes]
                    del new_classes[ci]
                    new_cnt = cnt[:]
                    new_f = f

        if new_f is None:
            continue
        if new_f < f:
            classes, cnt, f = new_classes, new_cnt, new_f
            if f < best_f:
                best_f = f
                best_classes = [c[:] for c in classes]
        else:
            delta = new_f - f
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                classes, cnt, f = new_classes, new_cnt, new_f
        T *= cool

    return best_classes, best_f


def construct():
    start = time.time()
    deadline = start + 120.0
    rng = random.Random(12345)

    asc = _MODS
    desc = list(reversed(_MODS))

    ca, ca_cnt, fa = _greedy_build(asc)
    cd, cd_cnt, fd = _greedy_build(desc)

    if fa <= fd:
        classes, cnt, f = ca, ca_cnt, fa
    else:
        classes, cnt, f = cd, cd_cnt, fd

    if f == 0:
        return classes

    best_classes, best_f = _local_search(classes, cnt, f, deadline, rng)
    return best_classes
