import time
import random

_DEADLINE = 110.0  # safety margin under 120s (evaluator kills at 150s)
_START = time.time()

def _elapsed():
    return time.time() - _START

def _all_diffs(s):
    # s sorted list
    out = []
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            out.append(s[j] - si)
    return out

def _valid(s):
    # s must contain 0, be distinct, sorted, all positive diffs distinct, and cover 1..30
    if 0 not in s:
        return False
    if len(s) != len(set(s)):
        return False
    d = _all_diffs(s)
    if len(d) != len(set(d)):
        return False
    sd = set(d)
    for k in range(1, 31):
        if k not in sd:
            return False
    return True

def _diameter(s):
    return max(s)

def _seed_set():
    # A known small Sidon set that covers 1..30 (Golomb ruler style).
    # We'll just use a construction and let local search fix it.
    # Start from a greedy Sidon set and then ensure coverage.
    # Use a randomized start.
    rng = random.Random(12345)
    a = [0]
    # build a Sidon set greedily with small numbers
    used = set()
    for d in range(1, 31):
        if d in used:
            continue
        # try to add x = v + d or v - d for some v in a
        placed = False
        for v in a:
            for cand in (v + d, v - d):
                if cand < 0 or cand in a:
                    continue
                trial = sorted(a + [cand])
                dd = _all_diffs(trial)
                if len(dd) == len(set(dd)):
                    a = trial
                    used = set(dd)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            # add a far element
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                dd = _all_diffs(trial)
                if len(dd) == len(set(dd)):
                    a = trial
                    used = set(dd)
                    break
                y += 1
    return a

def _local_search(s, rng, budget):
    # Hill-climbing: try to reduce diameter by moving an element to a smaller unused value
    # that keeps validity.
    cur = sorted(s)
    cur_diam = _diameter(cur)
    if not _valid(cur):
        # repair: not expected
        pass
    best = list(cur)
    best_diam = cur_diam
    while budget() > 0:
        improved = False
        n = len(cur)
        idx = rng.randrange(n)
        if idx == 0:
            continue  # keep 0 fixed
        val = cur[idx]
        # try moving val to a smaller candidate
        # candidate range: 1 .. val-1
        # sample a few candidates
        for _ in range(40):
            cand = rng.randrange(1, val)
            if cand in cur:
                continue
            trial = sorted(cur[:idx] + [cand] + cur[idx + 1:])
            if _valid(trial):
                nd = _diameter(trial)
                if nd < best_diam:
                    best = list(trial)
                    best_diam = nd
                    cur = list(trial)
                    improved = True
                    break
        # also try removing an element if >9 and still valid (reduces size, may allow smaller diam)
        if len(cur) > 9:
            idx2 = rng.randrange(1, len(cur))
            trial = sorted(cur[:idx2] + cur[idx2 + 1:])
            if _valid(trial):
                nd = _diameter(trial)
                if nd < best_diam:
                    best = list(trial)
                    best_diam = nd
                    cur = list(trial)
                    improved = True
        if not improved:
            # try adding a small element to enable later shrink (optional)
            pass
    return best, best_diam

def construct():
    global _START
    _START = time.time()
    deadline = _START + _DEADLINE
    def budget():
        return time.time() < deadline - 2.0

    # initial feasible
    s0 = _seed_set()
    if not _valid(s0):
        # fallback: brute small
        s0 = [0, 1, 4, 10, 12, 17, 20, 26, 28, 33]
    best = list(s0)
    best_diam = _diameter(best)

    rng = random.Random()
    # multiple restarts with perturbation
    while time.time() < deadline - 3.0:
        # perturb best
        cur = list(best)
        # random perturbation: move 1-3 elements to random small values
        for _ in range(rng.randint(1, 3)):
            idx = rng.randrange(1, len(cur))
            newv = rng.randrange(1, max(cur) + 20)
            if newv not in cur:
                cur[idx] = newv
        cur = sorted(cur)
        # local search from this start
        res, d = _local_search(cur, rng, budget)
        if d < best_diam:
            best = res
            best_diam = d
        # also try local search from best itself (continue improving)
        res2, d2 = _local_search(list(best), rng, budget)
        if d2 < best_diam:
            best = res2
            best_diam = d2

    # final safety: ensure validity; if not, return seed
    if not _valid(best):
        return s0
    return best
