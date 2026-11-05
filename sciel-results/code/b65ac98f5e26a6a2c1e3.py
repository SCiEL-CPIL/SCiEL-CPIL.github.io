"""Construct a 12-state binary deterministic synchronizing automaton with a long reset word.

Strategy:
  * Fast threshold via BFS on the subset automaton using integer bitmasks
    (12 states -> 12-bit masks) with precomputed image masks for both letters.
    This is ~10x faster than the frozenset BFS, letting us evaluate many candidates.
  * Start from the classic "cycle + monotone" family (a = 12-cycle, b monotone
    with one collision), then improve by systematic single-position / pair
    perturbations (a smooth surrogate for the min/max reset objective), plus a
    few structured families.
"""
import time
import random

N = 12
MASK_ALL = (1 << N) - 1

_DEADLINE = None

def _elapsed():
    return time.time() - _DEADLINE[0] if _DEADLINE else 0.0

def _time_left(limit):
    return (limit - 0.5) - _elapsed()

def _threshold(a, b, limit=None):
    """BFS on subset automaton (bitmasks). Returns shortest reset length or -1."""
    if limit is not None and _elapsed() > limit:
        return -1
    img_a = [0] * N
    img_b = [0] * N
    for i in range(N):
        img_a[i] = 1 << a[i]
        img_b[i] = 1 << b[i]
    visited = bytearray(1 << N)
    visited[MASK_ALL] = 1
    q = [MASK_ALL]
    head = 0
    depth = 0
    while head < len(q):
        n = len(q)
        for j in range(n):
            m = q[head]; head += 1
            # apply a
            ma = 0
            x = m
            while x:
                lsb = x & (-x)
                ma |= img_a[lsb.bit_length() - 1]
                x ^= lsb
            if not visited[ma]:
                if ma & (ma - 1) == 0:
                    return depth + 1
                visited[ma] = 1
                q.append(ma)
            # apply b
            mb = 0
            x = m
            while x:
                lsb = x & (-x)
                mb |= img_b[lsb.bit_length() - 1]
                x ^= lsb
            if not visited[mb]:
                if mb & (mb - 1) == 0:
                    return depth + 1
                visited[mb] = 1
                q.append(mb)
        depth += 1
        if limit is not None and _elapsed() > limit:
            return -1
    return -1

def _eval(a, b, limit):
    if _elapsed() > limit:
        return -1
    return _threshold(a, b, limit)

def _monotone_b():
    b = list(range(N - 1)) + [N - 2]
    return b

def _cycle_a():
    return [(i + 1) % N for i in range(N)]

def construct():
    global _DEADLINE
    _DEADLINE = [time.time()]
    budget = 80.0  # explicit safety margin before the 90s kill

    a = _cycle_a()
    b = _monotone_b()
    best_thr = _eval(a, b, budget)
    best_a, best_b = a[:], b[:]
    if best_thr <= 0:
        # guaranteed synchronizing fallback
        best_a = _cycle_a()
        best_b = [0] * N
        best_thr = _eval(best_a, best_b, budget)

    def consider(cand_a, cand_b):
        nonlocal best_thr, best_a, best_b
        if _elapsed() > budget:
            return
        thr = _eval(cand_a, cand_b, budget)
        if thr > best_thr:
            best_thr = thr
            best_a, best_b = cand_a[:], cand_b[:]

    # ---- Systematic neighborhood search over structured families ----
    # (a smooth surrogate for the min/max reset objective: local perturbations
    #  of the two transition maps, evaluated with the exact BFS.)

    # Family A: a = cycle, b = monotone, perturb b at one position.
    if _time_left(budget) > 0:
        for pos in range(N):
            if _time_left(budget) <= 0:
                break
            for val in range(N):
                if val == b[pos]:
                    continue
                bb = b[:]
                bb[pos] = val
                consider(a, bb)

    # Family A2: a = cycle, b = monotone, perturb b at two positions.
    if _time_left(budget) > 0:
        for p1 in range(N):
            if _time_left(budget) <= 0:
                break
            for p2 in range(p1 + 1, N):
                if _time_left(budget) <= 0:
                    break
                for v1 in range(N):
                    if _time_left(budget) <= 0:
                        break
                    for v2 in range(N):
                        bb = b[:]
                        bb[p1] = v1
                        bb[p2] = v2
                        consider(a, bb)

    # Family B: a = cycle, b = identity-with-one-defect.
    if _time_left(budget) > 0:
        for defect in range(N):
            if _time_left(budget) <= 0:
                break
            for target in range(N):
                if target == defect:
                    continue
                bb = list(range(N))
                bb[defect] = target
                consider(a, bb)

    # Family C: a = cycle, b = shift-by-step with one defect.
    if _time_left(budget) > 0:
        for step in range(1, N):
            if _time_left(budget) <= 0:
                break
            for defect in range(N):
                if _time_left(budget) <= 0:
                    break
                for target in range(N):
                    if _time_left(budget) <= 0:
                        break
                    bb = [(i + step) % N for i in range(N)]
                    bb[defect] = target
                    consider(a, bb)

    # Family D: a = cycle, b = "almost monotone" (monotone + one extra collision).
    if _time_left(budget) > 0:
        for pos in range(N):
            if _time_left(budget) <= 0:
                break
            for val in range(N):
                bb = _monotone_b()
                bb[pos] = val
                # ensure it is not exactly the base
                if bb == b:
                    continue
                consider(a, bb)

    # Family E: a = random permutation (a few), b = monotone.
    if _time_left(budget) > 0:
        rng = random.Random(12345)
        for _ in range(40):
            if _time_left(budget) <= 0:
                break
            aa = list(range(N))
            rng.shuffle(aa)
            consider(aa, b)

    # Final guaranteed-valid fallback.
    if best_thr <= 0:
        best_a = _cycle_a()
        best_b = [0] * N
        best_thr = _eval(best_a, best_b, budget)
        if best_thr <= 0:
            best_a = _cycle_a()
            best_b = [0] * N

    return best_a, best_b
