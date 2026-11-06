import time
from math import lcm
from functools import reduce

def construct():
    start = time.time()
    deadline = start + 88.0   # safety margin vs 90s kill

    L = reduce(lcm, range(1, 23))
    recip = {i: L // i for i in range(1, 23)}

    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_len = len(best)

    def time_up():
        return time.time() > deadline

    def can_add(sums, r):
        for s in sums:
            if s + r in sums:
                return False
        return True

    def spectral_gap(sums, r):
        m = None
        for s in sums:
            d = abs(r - s)
            if m is None or d < m:
                m = d
        return m if m is not None else 0

    def dfs(remaining, sums, chosen):
        nonlocal best, best_len
        if time_up():
            return
        if len(chosen) + len(remaining) <= best_len:
            return
        if not remaining:
            if len(chosen) > best_len:
                best = list(chosen)
                best_len = len(chosen)
            return

        scored = []
        for i in remaining:
            r = recip[i]
            if can_add(sums, r):
                scored.append((spectral_gap(sums, r), i, r))

        if not scored:
            # nothing can be added from remaining
            return

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        gap0, i0, r0 = scored[0]

        # Branch: include i0
        new_sums = sums | {s + r0 for s in sums}
        chosen.append(i0)
        rem2 = [x for x in remaining if x != i0]
        dfs(rem2, new_sums, chosen)
        chosen.pop()
        if time_up():
            return
        # Branch: exclude i0
        rem3 = [x for x in remaining if x != i0]
        dfs(rem3, sums, chosen)

    def grow(sums, chosen):
        nonlocal best, best_len
        changed = True
        while changed and not time_up():
            changed = False
            scored = []
            for i in range(1, 23):
                if i in chosen:
                    continue
                r = recip[i]
                if can_add(sums, r):
                    scored.append((spectral_gap(sums, r), i, r))
            if not scored:
                break
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            _, i0, r0 = scored[0]
            sums = sums | {s + r0 for s in sums}
            chosen.append(i0)
            changed = True
            if len(chosen) > best_len:
                best = list(chosen)
                best_len = len(chosen)
        return chosen, sums

    def build_sums(chosen):
        sums = {0}
        for i in chosen:
            r = recip[i]
            sums = sums | {s + r for s in sums}
        return sums

    # Main heuristic DFS
    dfs(list(range(1, 23)), {0}, [])

    # Local grow from current best
    if not time_up():
        chosen = list(best)
        sums = build_sums(chosen)
        grow(sums, chosen)

    # Local grow from a couple of seeds
    for seed in ([1, 2, 3, 5, 7, 11, 13, 17, 19]):
        if time_up():
            break
        chosen = [i for i in seed if 1 <= i <= 22]
        sums = {0}
        ok = True
        for i in chosen:
            r = recip[i]
            if not can_add(sums, r):
                ok = False
                break
            sums = sums | {s + r for s in sums}
        if not ok:
            continue
        grow(sums, chosen)

    return best
