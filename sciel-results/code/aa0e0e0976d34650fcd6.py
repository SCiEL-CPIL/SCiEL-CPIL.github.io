"""Improved Sidon-set constructor: max-coverage greedy + local-search polish."""
import time

MAXD = 30
MAXELEM = 32
BOUND = 10**6


def construct():
    start = time.time()
    DEADLINE = start + 120.0
    SAFETY = 5.0

    best = None
    best_max = BOUND

    def is_valid(a):
        if len(a) != len(set(a)):
            return False
        if len(a) > MAXELEM:
            return False
        if 0 not in a:
            return False
        dset = set()
        for i in range(len(a)):
            ai = a[i]
            for j in range(i + 1, len(a)):
                d = a[j] - ai
                if d in dset:
                    return False
                dset.add(d)
        for d in range(1, MAXD + 1):
            if d not in dset:
                return False
        return True

    def diff_set(a):
        s = set()
        for i in range(len(a)):
            ai = a[i]
            for j in range(i + 1, len(a)):
                s.add(a[j] - ai)
        return s

    def coverage(a, x):
        # number of missing diffs (1..30) realized by adding x, assuming Sidon stays valid
        ds = diff_set(a)
        missing = [d for d in range(1, MAXD + 1) if d not in ds]
        cnt = 0
        for d in missing:
            if (x - d in a) or (x + d in a):
                cnt += 1
        return cnt

    def sidon_ok(a, x):
        # check adding x preserves distinctness of positive differences
        for y in a:
            d = abs(x - y)
            if d == 0:
                return False
            # this difference d must not already exist in a
            # but we need to know existing diffs; do a full check for correctness
        # full check
        trial = a + [x]
        seen = set()
        for i in range(len(trial)):
            ai = trial[i]
            for j in range(i + 1, len(trial)):
                d = trial[j] - ai
                if d in seen:
                    return False
                seen.add(d)
        return True

    # ---- Greedy max-coverage build ----
    def greedy_build(budget):
        a = [0]
        while True:
            if time.time() > DEADLINE - SAFETY:
                break
            ds = diff_set(a)
            missing = [d for d in range(1, MAXD + 1) if d not in ds]
            if not missing:
                break
            if len(a) >= MAXELEM:
                break
            # candidate pool: near each existing element (±d for missing d) and small fresh
            cands = set()
            for v in a:
                for d in missing:
                    for s in (d, -d):
                        w = v + s
                        if 0 <= w < budget and w not in a:
                            cands.add(w)
            # also some small fresh candidates
            lo = min(a)
            hi = max(a)
            for w in range(0, max(hi + 1, 1)):
                if w not in a:
                    cands.add(w)
            # rank candidates: by coverage desc, then by resulting max asc
            scored = []
            for w in cands:
                if not sidon_ok(a, w):
                    continue
                c = coverage(a, w)
                if c <= 0:
                    continue
                new_max = max(hi, w)
                scored.append((c, new_max, w))
            if not scored:
                break
            scored.sort(key=lambda t: (-t[0], t[1]))
            best_c, best_m, best_w = scored[0]
            a.append(best_w)
        return sorted(a)

    # ---- Local search polish ----
    def polish(a):
        nonlocal best, best_max
        a = sorted(a)
        cur_max = max(a)
        if cur_max < best_max and is_valid(a):
            best = list(a)
            best_max = cur_max
        # repeated improvement passes
        while time.time() < DEADLINE - SAFETY:
            improved = False
            n = len(a)
            # try moving each element down to a smaller unused value preserving validity
            for i in range(n):
                if time.time() > DEADLINE - SAFETY:
                    break
                orig = a[i]
                if orig == 0:
                    continue
                # candidate smaller positions
                cands = set()
                hi = max(a)
                for v in range(0, orig):
                    if v not in a:
                        cands.add(v)
                # also consider positions near existing to gain coverage (but we only need coverage already)
                # sort by resulting max asc
                for v in sorted(cands):
                    trial = list(a)
                    trial[i] = v
                    trial.sort()
                    if not is_valid(trial):
                        continue
                    tm = max(trial)
                    if tm < cur_max:
                        a = trial
                        cur_max = tm
                        improved = True
                        if cur_max < best_max:
                            best = list(a)
                            best_max = cur_max
                        break
            if not improved:
                break
        return a

    # Run multiple greedy builds with varying budgets, then polish
    budgets = [40, 50, 60, 80, 100, 130, 160, 200]
    for b in budgets:
        if time.time() > DEADLINE - SAFETY:
            break
        a = greedy_build(b)
        if not a:
            continue
        # ensure validity; if not fully covering, we can still polish
        polish(a)

    # Final safety: if nothing valid yet, fall back to a guaranteed valid set
    if best is None or not is_valid(best):
        # simple guaranteed construction (seed-like) as fallback
        a = [0]
        for d in range(1, MAXD + 1):
            ds = diff_set(a)
            if d in ds:
                continue
            placed = False
            for v in a:
                for s in (d, -d):
                    w = v + s
                    if w >= 0 and w not in a and sidon_ok(a, w):
                        a.append(w)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                y = 2 * max(a) + 3 * d + 1
                while not sidon_ok(a, y) or not sidon_ok(a + [y], y + d):
                    y += 1
                a.append(y)
                a.append(y + d)
        a = sorted(a)
        if is_valid(a):
            best = a
            best_max = max(a)

    if best is None:
        # ultimate fallback: a known-valid construction via large gaps
        a = [0]
        gap = 1
        for d in range(1, MAXD + 1):
            while not sidon_ok(a, gap):
                gap += 1
            a.append(gap)
            gap += 1
        return sorted(a)

    return sorted(best)
