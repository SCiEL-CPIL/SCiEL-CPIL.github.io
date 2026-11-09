"""Sidon set containing 0 with differences 1..30, minimizing max element."""
import random
import time

# EVOLVE-BLOCK-START
import time as _time

def construct():
    deadline = _time.time() + 140  # safety margin below 150s
    random.seed(12345)

    def all_diffs(s):
        # returns list of positive differences
        n = len(s)
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                out.append(s[j] - s[i])
        return out

    def check_sidon(s):
        d = all_diffs(s)
        return len(set(d)) == len(d)

    def missing_diffs(s):
        d = set(all_diffs(s))
        return [i for i in range(1, 31) if i not in d]

    def objective(s):
        # lower is better: primary = missing count, secondary = max element
        miss = len(missing_diffs(s))
        return (miss, max(s))

    # Start from a known good-ish construction: use a Sidon set that covers 1..30
    # Build via randomized search.

    def random_init():
        # Start with 0 and add random elements to cover diffs, keeping Sidon
        s = [0]
        used = set()
        attempts = 0
        while len(missing_diffs(s)) > 0 and attempts < 20000:
            attempts += 1
            # pick a missing diff and try to realize it
            miss = missing_diffs(s)
            d = random.choice(miss)
            # try adding a new element x such that x - y = d or y - x = d for some y in s
            cand = []
            for y in s:
                cand.append(y + d)
                cand.append(y - d)
            random.shuffle(cand)
            for x in cand:
                if x < 0 or x in s:
                    continue
                trial = sorted(s + [x])
                if check_sidon(trial):
                    s = trial
                    break
        return s

    def local_search(s):
        # Try to reduce max element by moving elements down, or removing redundant
        best = s[:]
        best_obj = objective(best)
        for _ in range(200000):
            if _time.time() > deadline - 2:
                break
            i = random.randrange(len(best))
            old = best[i]
            # try to move element to a smaller value
            lo = 0
            hi = old - 1
            if hi < lo:
                continue
            # bias toward smaller values
            new = random.randrange(lo, hi + 1) if hi > lo else lo
            if new == old:
                continue
            trial = best[:]
            trial[i] = new
            if len(set(trial)) != len(trial):
                continue
            trial.sort()
            if not check_sidon(trial):
                continue
            obj = objective(trial)
            if obj < best_obj:
                best_obj = obj
                best = trial
            else:
                # occasionally accept worse to escape (rare)
                if random.random() < 0.001 and obj[0] == best_obj[0]:
                    best = trial
                    best_obj = obj
        return best

    def try_remove(s):
        # remove any element that is not needed for a difference
        changed = True
        while changed:
            changed = False
            for i in range(len(s)):
                if s[i] == 0:
                    continue
                trial = s[:i] + s[i+1:]
                if check_sidon(trial) and set(missing_diffs(trial)) == set(missing_diffs(s)):
                    # removing doesn't lose any diff
                    if max(trial) < max(s):
                        s = trial
                        changed = True
                        break
        return s

    # Generate multiple random inits and improve
    best = None
    best_obj = None
    n_inits = 0
    while _time.time() < deadline - 5:
        n_inits += 1
        s = random_init()
        s = try_remove(s)
        s = local_search(s)
        s = try_remove(s)
        obj = objective(s)
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best = s
        # If we have full coverage and a low max, try to polish
        if best_obj[0] == 0:
            # fully covered; try to shrink further
            polished = local_search(best)
            obj2 = objective(polished)
            if obj2 < best_obj:
                best_obj = obj2
                best = polished

    # Final safety: ensure validity
    if best is None:
        # fallback greedy
        a = [0]
        for d in range(1, 31):
            if d not in set(all_diffs(sorted(a))):
                for x in sorted(set(v + d for v in a) | set(v - d for v in a)):
                    if x < 0:
                        continue
                    trial = sorted(a + [x])
                    if check_sidon(trial):
                        a = trial
                        break
                else:
                    y = 2 * max(a) + 3 * d + 1
                    while True:
                        trial = sorted(a + [y, y + d])
                        if check_sidon(trial):
                            a = trial
                            break
                        y += 1
        best = a

    assert 0 in best
    assert len(set(best)) == len(best)
    assert max(best) <= 10**6
    assert check_sidon(best)
    assert len(best) <= 32
    assert set(missing_diffs(best)) == set()
    return best
# EVOLVE-BLOCK-END
