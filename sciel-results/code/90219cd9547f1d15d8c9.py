import time
import sys

_START = None
_DEADLINE = 110.0  # safety margin under the 120 s wall-clock limit
FULL30 = (1 << 30) - 1  # bits 0..29 represent differences 1..30


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _popcount(x):
    return bin(x).count("1")


def _is_valid(S):
    """Defensive check: distinct ints, contains 0, size<=32, <=1e6, Sidon, covers 1..30."""
    if 0 not in S:
        return False
    if len(S) != len(set(S)):
        return False
    if len(S) > 32:
        return False
    if max(S) > 10**6:
        return False
    diffs = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            d = abs(S[i] - S[j])
            if d in diffs:
                return False
            diffs.add(d)
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True


def _greedy_baseline():
    """Always returns a valid Sidon set covering 1..30 (guaranteed progress)."""
    C = [0]
    used = set()
    missing = set(range(1, 31))
    while missing:
        d = min(missing)
        last = C[-1]
        chosen = None
        # Try a single new element x = s + d (s in C) that keeps the set Sidon.
        for x in sorted({s + d for s in C}):
            if x <= last:
                continue
            if all((x - s) not in used for s in C):
                chosen = x
                break
        if chosen is None:
            # Fallback: add a far-out pair (T, T+d) that realizes d without collisions.
            T = last + 1
            while True:
                ok = True
                for s in C:
                    if (T - s) in used or (T + d - s) in used:
                        ok = False
                        break
                if ok and T + d <= 10**6:
                    break
                T += 1
            for s in C:
                used.add(T - s)
                used.add(T + d - s)
            used.add(d)
            missing.discard(d)
            for dd in list(range(1, 31)):
                if dd in used:
                    missing.discard(dd)
            C = sorted(C + [T, T + d])
            continue
        # Commit the single element.
        for s in C:
            used.add(chosen - s)
        C.append(chosen)
        for dd in list(range(1, 31)):
            if dd in used:
                missing.discard(dd)
    return C


def construct():
    global _START
    _START = time.time()
    sys.setrecursionlimit(10000)

    baseline = _greedy_baseline()
    best = [baseline]
    best_m = [max(baseline)]

    FULL = FULL30

    def dfs(C, used, cov):
        if _time_left() < 3.0:
            return
        if cov == FULL:
            m = C[-1]
            if m < best_m[0]:
                best_m[0] = m
                best[0] = list(C)
                if m == 30:  # theoretical minimum (30 must be a difference, 0 in set)
                    raise _Stop()
            return
        if len(C) >= 32:
            return
        # Optimistic coverage bound: can remaining slots cover all missing differences?
        missing = FULL & ~cov
        mc = _popcount(missing)
        k = 32 - len(C)
        maxnew = k * len(C) + k * (k - 1) // 2
        if mc > maxnew:
            return
        last = C[-1]
        cap = best_m[0]  # any new element >= cap cannot improve the diameter

        cand = set()
        # Candidates that realize a still-missing difference d via x = s + d.
        for d in range(1, 31):
            if (cov >> (d - 1)) & 1:
                continue
            for s in C:
                x = s + d
                if last < x < cap:
                    cand.add(x)
        # Small bridge candidates for completeness.
        for b in range(1, 4):
            x = last + b
            if x < cap:
                cand.add(x)

        scored = []
        for x in cand:
            dm = 0
            nc = 0
            ok = True
            for s in C:
                dd = x - s
                if (used >> dd) & 1:
                    ok = False
                    break
                dm |= (1 << dd)
                if 1 <= dd <= 30 and not ((cov >> (dd - 1)) & 1):
                    nc += 1
            if not ok:
                continue
            scored.append((-nc, x, dm, nc))
        scored.sort()
        for _neg, x, dm, nc in scored[:60]:
            # new coverage bits from this element
            nb = cov
            for s in C:
                dd = x - s
                if 1 <= dd <= 30:
                    nb |= (1 << (dd - 1))
            dfs(C + [x], used | dm, nb)

    class _Stop(Exception):
        pass

    try:
        dfs([0], 0, 0)
    except _Stop:
        pass

    result = best[0]
    if not _is_valid(result):
        result = baseline
        if not _is_valid(result):
            # ultimate defensive fallback: re-run greedy
            result = _greedy_baseline()
    return result
