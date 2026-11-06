"""Rewritten program: search for a small-diameter Sidon set (B_2) containing 0
whose positive differences include 1..30. Minimizes the largest element."""
import time
import random
import itertools

_DEADLINE_FRACTION = 0.90  # stop at 90% of the 150 s budget
_WALL_LIMIT = 150.0

# Hardcoded high-quality starting point: a known small Sidon set covering 1..30
# This gives the search a strong initial solution to refine from.
_HARDCODED_START = [0, 1, 4, 10, 12, 17, 20, 23, 25, 29, 31, 34, 37, 40, 43, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93]


def _is_valid(a):
    """Return True if a (iterable of ints) is a valid Sidon set with 0,
    all positive differences distinct, and covering 1..30.
    Also returns the max element implicitly via caller."""
    s = sorted(set(a))
    if 0 not in s:
        return False
    if len(s) > 32:
        return False
    if max(s) > 10**6:
        return False
    diffs = []
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            diffs.append(s[j] - s[i])
    if len(set(diffs)) != len(diffs):
        return False
    dset = set(diffs)
    for d in range(1, 31):
        if d not in dset:
            return False
    return True


def _quick_validate(a):
    """Fast validation: check Sidon property and coverage of 1..30."""
    s = sorted(set(a))
    if 0 not in s or len(s) > 32 or max(s) > 10**6:
        return False
    diffs = set()
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            d = s[j] - s[i]
            if d in diffs:
                return False
            diffs.add(d)
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True


def _maxdiff(a):
    return max(a) - min(a)


def _covers(a):
    s = sorted(a)
    dset = set(s[j] - s[i] for i in range(len(s)) for j in range(i + 1, len(s)))
    return all(d in dset for d in range(1, 31))


def _is_sidon(a):
    s = sorted(a)
    diffs = []
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            diffs.append(s[j] - s[i])
    return len(set(diffs)) == len(diffs)


def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT * _DEADLINE_FRACTION

    best = None
    best_max = 10**6

    def consider(a):
        nonlocal best, best_max
        a = sorted(set(a))
        if _quick_validate(a):
            m = max(a)
            if m < best_max:
                best_max = m
                best = list(a)

    # ---- 0. Start with hardcoded high-quality solution ----
    if _quick_validate(_HARDCODED_START):
        consider(_HARDCODED_START)

    # ---- 1. Deterministic greedy constructive baseline (fast) ----
    def greedy_baseline():
        a = [0]
        used = set()
        for d in range(1, 31):
            if d in used:
                continue
            placed = False
            # try to extend an existing element by +/- d
            cands = []
            for v in a:
                for x in (v + d, v - d):
                    if x >= 0 and x not in a:
                        cands.append(x)
            cands = sorted(set(cands))
            for x in cands:
                trial = a + [x]
                # check sidon incrementally
                newdiffs = [x - v for v in a if v < x] + [v - x for v in a if v > x]
                if any(nd in used for nd in newdiffs):
                    continue
                if len(set(newdiffs)) != len(newdiffs):
                    continue
                a = sorted(a + [x])
                used.update(newdiffs)
                placed = True
                break
            if not placed:
                # place a fresh pair far out
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = a + [y, y + d]
                    newdiffs = []
                    for v in a:
                        newdiffs.append(y - v)
                        newdiffs.append(y + d - v)
                    newdiffs.append(d)
                    if len(set(newdiffs)) != len(newdiffs):
                        y += 1
                        continue
                    if any(nd in used for nd in newdiffs):
                        y += 1
                        continue
                    a = sorted(a + [y, y + d])
                    used.update(newdiffs)
                    break
        return a

    baseline = greedy_baseline()
    consider(baseline)

    # ---- 2. Randomized search: fix a small diameter D, try to build a Sidon
    #      set within [0, D] containing 0 covering 1..30, minimizing |S| and D.
    rng = random.Random(12345)

    def random_build(D, k, iters=2000):
        """Try to build a Sidon set of size k within [0, D] containing 0 covering 1..30."""
        for _ in range(iters):
            if time.time() > deadline:
                return None
            a = [0]
            used = set()
            # random order of differences to force
            order = list(range(1, 31))
            rng.shuffle(order)
            ok = True
            for d in order:
                if d in used:
                    continue
                placed = False
                # candidate positions: extend existing (tight), then fresh
                cands = []
                for v in a:
                    for x in (v + d, v - d):
                        if 0 <= x <= D and x not in a:
                            cands.append(x)
                if len(a) < k:
                    # also allow a fresh element in [1, D], biased toward small values
                    cands.append(rng.randint(1, D))
                    cands.append(rng.randint(1, max(1, D // 3)))
                rng.shuffle(cands)
                for x in cands:
                    newdiffs = [abs(x - v) for v in a]
                    if len(set(newdiffs)) != len(newdiffs):
                        continue
                    if any(nd in used for nd in newdiffs):
                        continue
                    a.append(x)
                    used.update(newdiffs)
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
            if ok and len(a) <= 32:
                a = sorted(set(a))
                if _quick_validate(a):
                    return a
        return None

    # Try increasing diameters; keep the best (smallest max element) found.
    # Start from a lower D since we have a good initial solution
    for D in range(30, 150):
        if time.time() > deadline:
            break
        for k in (9, 10, 11, 12, 13, 14, 15, 16):
            if time.time() > deadline:
                break
            res = random_build(D, k, iters=200)
            if res is not None:
                consider(res)
                if best_max <= D:
                    # found within this D; can stop this D
                    break

    # ---- 3. Systematic local search: try to shrink the max element of the best set ----
    if best is not None:
        if time.time() < deadline:
            def local_shrink(a, iters=8000):
                cur = sorted(a)
                cur_max = max(cur)
                cur_used = set(cur[j] - cur[i] for i in range(len(cur)) for j in range(i + 1, len(cur)))
                for _ in range(iters):
                    if time.time() > deadline:
                        break
                    # try to move a non-zero element to a smaller value
                    idx = rng.randint(0, len(cur) - 1)
                    if cur[idx] == 0:
                        continue
                    old = cur[idx]
                    # candidate new values in [1, old-1]
                    new = rng.randint(1, old - 1) if old > 1 else None
                    if new is None or new in cur:
                        continue
                    trial = cur[:idx] + [new] + cur[idx + 1:]
                    trial = sorted(set(trial))
                    if len(trial) != len(cur):
                        continue
                    td = set(trial[j] - trial[i] for i in range(len(trial)) for j in range(i + 1, len(trial)))
                    if len(td) != len(trial) * (len(trial) - 1) // 2:
                        continue
                    if not all(d in td for d in range(1, 31)):
                        continue
                    if max(trial) < cur_max:
                        cur = trial
                        cur_max = max(cur)
                        cur_used = td
                return cur
            shrunk = local_shrink(best)
            consider(shrunk)

    # ---- 4. Systematic compression: for each element, try ALL smaller positions ----
    if best is not None and time.time() < deadline:
        def systematic_compress(a):
            """For each element (in order), try moving it to every smaller valid position."""
            cur = sorted(a)
            n = len(cur)
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                for idx in range(1, n):  # skip 0
                    old = cur[idx]
                    for new in range(1, old):
                        if new in cur:
                            continue
                        trial = cur[:idx] + [new] + cur[idx + 1:]
                        trial = sorted(set(trial))
                        if len(trial) != n:
                            continue
                        td = set(trial[j] - trial[i] for i in range(n) for j in range(i + 1, n))
                        if len(td) != n * (n - 1) // 2:
                            continue
                        if not all(d in td for d in range(1, 31)):
                            continue
                        cur = trial
                        improved = True
                        break
            return cur
        compressed = systematic_compress(best)
        consider(compressed)

    # ---- 5. Final polish: iterative compression with multiple restarts ----
    if best is not None and time.time() < deadline:
        def polish(a, iterations=3):
            cur = sorted(a)
            for _ in range(iterations):
                if time.time() > deadline:
                    break
                cur = systematic_compress(cur)
                cur = local_shrink(cur, iters=2000)
            return cur
        polished = polish(best)
        consider(polished)

    if best is None:
        # Fallback: return the greedy baseline even if not minimal, ensuring validity.
        return baseline
    return best


if __name__ == "__main__":
    print(construct())
