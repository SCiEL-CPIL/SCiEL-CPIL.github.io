# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_p = 7
_q = 49
_total = _p ** 4  # 2401
_DEADLINE = 540.0  # safety margin under 630s wall-clock
_start = time.time()


def _remaining_budget():
    return _DEADLINE - (time.time() - _start)


def _encode(xa, xb, ya, yb):
    return (xa * _p + xb) * _q + (ya * _p + yb)


def _decode(x):
    y_part = x % _q
    x_part = x // _q
    xa = x_part // _p
    xb = x_part % _p
    ya = y_part // _p
    yb = y_part % _p
    return xa, xb, ya, yb


def _build_lines():
    """lines[x][d] = frozenset of the other 48 points on direction d through x.
    d=0 vertical (0,1); d=1..49 slope m=d-1."""
    lines = [None] * _total
    for xa in range(_p):
        for xb in range(_p):
            for ya in range(_p):
                for yb in range(_p):
                    x = _encode(xa, xb, ya, yb)
                    dirs = []
                    # direction 0: vertical (0,1)
                    s = set()
                    for t in range(_q):
                        if t == 0:
                            continue
                        ta, tb = t // _p, t % _p
                        nxa, nxb = xa, xb
                        nya, nyb = (ya + ta) % _p, (yb + tb) % _p
                        s.add(_encode(nxa, nxb, nya, nyb))
                    dirs.append(frozenset(s))
                    # directions 1..49: slope m
                    for m in range(_q):
                        ma, mb = m // _p, m % _p
                        s = set()
                        for t in range(_q):
                            if t == 0:
                                continue
                            ta, tb = t // _p, t % _p
                            pma = (ta * ma + tb * mb * 3) % _p
                            pmb = (ta * mb + tb * ma) % _p
                            nxa = (xa + pma) % _p
                            nxb = (xb + pmb) % _p
                            nya = (ya + ta) % _p
                            nyb = (yb + tb) % _p
                            s.add(_encode(nxa, nxb, nya, nyb))
                        dirs.append(frozenset(s))
                    lines[x] = dirs
    return lines


def _check_validity(N_set, lines):
    for x in range(_total):
        ok = False
        for d in range(50):
            if lines[x][d] <= N_set:
                ok = True
                break
        if not ok:
            return False
    return True


def _greedy(lines, order):
    in_N = [True] * _total
    complement = set()
    for x in order:
        if x in complement:
            continue
        for d in range(50):
            L = lines[x][d]
            if all(in_N[y] for y in L):
                in_N[x] = False
                complement.add(x)
                break
    return complement


def _local_search(lines, complement, max_iters, seed):
    rng = random.Random(seed)
    comp = set(complement)
    in_N = [x not in comp for x in range(_total)]

    def can_add(x):
        if x in comp:
            return False
        for d in range(50):
            L = lines[x][d]
            if all(in_N[y] for y in L):
                return True
        return False

    # Greedily add any addable
    changed = True
    while changed:
        changed = False
        for x in range(_total):
            if x in comp:
                continue
            if can_add(x):
                comp.add(x)
                in_N[x] = False
                changed = True

    for it in range(max_iters):
        if _remaining_budget() < 30:
            break
        x = rng.randrange(_total)
        if x in comp:
            continue
        for d in range(50):
            L = lines[x][d]
            blockers = [y for y in L if y in comp]
            if not blockers:
                comp.add(x)
                in_N[x] = False
                break
            elif len(blockers) == 1:
                y = blockers[0]
                comp.discard(y)
                in_N[y] = True
                if can_add(x):
                    comp.add(x)
                    in_N[x] = False
                else:
                    comp.add(y)
                    in_N[y] = False
                break
    return comp


def nikodym_set() -> np.ndarray:
    lines = _build_lines()

    best_comp = set()
    best_size = -1

    n_greedy = 6
    for g in range(n_greedy):
        if _remaining_budget() < 120:
            break
        order = list(range(_total))
        random.shuffle(order)
        comp = _greedy(lines, order)
        if len(comp) > best_size:
            best_size = len(comp)
            best_comp = set(comp)

    comp = _local_search(lines, best_comp, max_iters=15000, seed=999)
    if len(comp) > best_size:
        best_size = len(comp)
        best_comp = set(comp)

    N_set = set(range(_total)) - best_comp

    if not _check_validity(N_set, lines):
        N_set = set(range(_total))

    pts = []
    for x in sorted(N_set):
        xa, xb, ya, yb = _decode(x)
        pts.append([[xa, xb], [ya, yb]])

    return np.array(pts, dtype=np.int64)


# EVOLVE-BLOCK-END
