"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def _positive_diffs(s):
    n = len(s)
    return [s[j] - s[i] for i in range(n) for j in range(i + 1, n)]

def _is_sidon(s):
    d = _positive_diffs(s)
    return len(d) == len(set(d))

def _covers_1_to_30(s):
    d = set(_positive_diffs(s))
    return all(x in d for x in range(1, 31))

def _valid(s):
    if len(s) < 1 or len(s) > 32:
        return False
    if 0 not in s:
        return False
    if max(s) > 10**6:
        return False
    if not _is_sidon(s):
        return False
    if not _covers_1_to_30(s):
        return False
    return True

def _greedy_construct():
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

def _random_local_search():
    best = None
    best_val = 10**7
    rng = random.Random(12345)
    # Start from a few random seeds
    for attempt in range(200):
        if time.time() > _DEADLINE:
            break
        # random initial set containing 0, size 9-12
        size = rng.randint(9, 12)
        upper = rng.randint(60, 200)
        s = {0}
        while len(s) < size:
            s.add(rng.randint(1, upper))
        s = sorted(s)
        if not _is_sidon(s):
            continue
        cur = s
        cur_val = max(cur)
        for step in range(3000):
            if time.time() > _DEADLINE:
                break
            # mutate: add, remove, or shift an element
            op = rng.random()
            if op < 0.4 and len(cur) < 32:
                # add a random element
                cand = cur + [rng.randint(1, max(cur) + 50)]
                cand = sorted(cand)
                if len(set(cand)) == len(cand) and _is_sidon(cand):
                    cur = cand
            elif op < 0.7 and len(cur) > 9:
                # remove a random non-zero element
                idx = rng.randint(1, len(cur) - 1)
                cand = cur[:idx] + cur[idx + 1:]
                if _is_sidon(cand):
                    cur = cand
            else:
                # shift a random non-zero element
                idx = rng.randint(1, len(cur) - 1)
                delta = rng.randint(-10, 10)
                new_val = cur[idx] + delta
                if new_val <= 0:
                    continue
                cand = cur[:idx] + [new_val] + cur[idx + 1:]
                cand = sorted(cand)
                if len(set(cand)) == len(cand) and _is_sidon(cand):
                    cur = cand
            # check if covers 1..30
            if _covers_1_to_30(cur):
                val = max(cur)
                if val < best_val:
                    best_val = val
                    best = cur[:]
    return best

def _greedy_with_pruning():
    """Greedy but try to keep the set compact by choosing the smallest valid x."""
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        # Try adding a single element x such that x - v = d or v - x = d for some v in a
        # and the result is Sidon. Prefer smallest max.
        candidates = []
        for v in a:
            for x in (v + d, v - d):
                if x >= 0 and x not in a:
                    candidates.append(x)
        candidates = sorted(set(candidates))
        for x in candidates:
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            # Add two elements to realize d
            y = max(a) + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

def construct():
    t0 = time.time()
    # Try greedy first (fast, gives a valid baseline)
    g = _greedy_construct()
    best = g
    best_val = max(g)
    # Try improved greedy
    g2 = _greedy_with_pruning()
    if _valid(g2) and max(g2) < best_val:
        best = g2
        best_val = max(g2)
    # Try local search to improve
    ls = _random_local_search()
    if ls is not None and max(ls) < best_val:
        best = ls
        best_val = max(ls)
    return best
# EVOLVE-BLOCK-END
