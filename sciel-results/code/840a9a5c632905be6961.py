"""Sidon set containing 0, covering differences 1..30, minimizing the largest element."""
import time
import random

START = time.time()
DEADLINE = 115.0  # safety margin before the 120s kill (evaluator wall-clock)

def _time_left():
    return DEADLINE - time.time()

def _is_valid(a):
    """All positive differences distinct AND every d in 1..30 present."""
    seen = set()
    for i in range(len(a)):
        ai = a[i]
        for j in range(i + 1, len(a)):
            d = a[j] - ai
            if d > 30:
                break
            if d in seen:
                return False
            seen.add(d)
    return len(seen) == 30

def _diameter(a):
    return a[-1]

def _dfs(a, used, missing, depth, best):
    """Backtrack to complete a valid set. Returns (found, best) tuple."""
    if not missing:
        return True, a
    if depth > 25:
        return False, best
    if len(a) > 32:
        return False, best
    # choose the most constrained (largest) missing difference
    d = max(missing)
    a_sorted = sorted(a)
    # candidate elements: x such that x - a_i == d for some a_i in a (x > a_i)
    # i.e. x = a_i + d for some a_i
    cands = set()
    for ai in a_sorted:
        x = ai + d
        if x <= a_sorted[-1]:
            # x must be new and > all? not necessarily; but to keep sorted, x can be anywhere
            pass
        cands.add(x)
        cands.add(x - d)  # also a_i - d if >= 0 (x = a_i, a_j = a_i + d already in set? no)
    # Better: to realize difference d, need two elements p<q with q-p=d.
    # If neither is in set, add both; if one is, add the other.
    # We try adding a single new element x that pairs with an existing element.
    options = set()
    for ai in a_sorted:
        options.add(ai + d)   # new upper element
        if ai - d >= 0:
            options.add(ai - d)  # new lower element
    # sort by resulting diameter (smaller max first)
    options = sorted(options, key=lambda x: max(x, a_sorted[-1]))
    for x in options:
        if x in used:
            continue
        if x < 0:
            continue
        new_a = a_sorted + [x]
        new_a.sort()
        # check no duplicate differences introduced
        ok = True
        for i in range(len(new_a)):
            for j in range(i + 1, len(new_a)):
                dd = new_a[j] - new_a[i]
                if dd > 30:
                    break
                # we can't easily track per-diff count here without a set; do full check
        # full validation of partial set for duplicates
        seen = set()
        for i in range(len(new_a)):
            for j in range(i + 1, len(new_a)):
                dd = new_a[j] - new_a[i]
                if dd in seen:
                    ok = False
                    break
                seen.add(dd)
            if not ok:
                break
        if not ok:
            continue
        new_missing = missing - {d}
        # also remove any other differences now covered
        for dd in list(new_missing):
            if dd in seen:
                new_missing.discard(dd)
        new_used = used | {x}
        found, best2 = _dfs(new_a, new_used, new_missing, depth + 1, best)
        if found:
            return True, best2
        if best2 is not None and (best is None or _diameter(best2) < _diameter(best)):
            best = best2
    return False, best

def _initial_dfs():
    """Deterministic backtracking from a small seed to get a valid solution."""
    # Start with 0 and a few small elements to cover small differences.
    # We'll try to build up greedily with backtracking.
    a = [0]
    used = {0}
    missing = set(range(1, 31))
    found, best = _dfs(a, used, missing, 0, None)
    if found:
        return best
    # Fallback: greedy
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        placed = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                placed = True
                break
        if not placed:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

def _sa_refine(start_set, max_iters):
    """Simulated Annealing to reduce diameter while keeping validity."""
    cur = sorted(start_set)
    cur_diam = _diameter(cur)
    best = list(cur)
    best_diam = cur_diam
    T = 20.0
    T_min = 0.01
    cooling = 0.9995
    rng = random.Random(42)

    def valid(a):
        return _is_valid(a)

    it = 0
    while it < max_iters and _time_left() > 5.0:
        it += 1
        a = list(cur)
        # random move: pick an element, perturb it
        idx = rng.randrange(len(a))
        delta = rng.randint(-3, 3)
        new_val = a[idx] + delta
        if new_val < 0 or new_val > 10**6:
            continue
        # ensure distinct
        if new_val in a:
            continue
        a[idx] = new_val
        a.sort()
        if not valid(a):
            continue
        new_diam = _diameter(a)
        delta_diam = new_diam - cur_diam
        if delta_diam <= 0 or rng.random() < pow(2.718281828, -delta_diam / max(T, 1e-9)):
            cur = a
            cur_diam = new_diam
            if new_diam < best_diam:
                best_diam = new_diam
                best = list(a)
        T *= cooling
        if T < T_min:
            T = T_min
    return best

def construct():
    t0 = time.time()
    # Phase 1: deterministic DFS for a good initial solution
    initial = _initial_dfs()
    if not _is_valid(initial):
        # Should not happen; return greedy anyway
        return initial

    best = list(initial)
    best_diam = _diameter(best)

    # Phase 2: Simulated Annealing to refine
    remaining = _time_left()
    if remaining > 10.0:
        max_iters = int(remaining * 2000)  # rough budget
        refined = _sa_refine(best, max_iters)
        if _is_valid(refined) and _diameter(refined) < best_diam:
            best = refined
            best_diam = _diameter(refined)

    # Final safety: ensure we return a valid set
    if not _is_valid(best):
        best = initial
    return best
