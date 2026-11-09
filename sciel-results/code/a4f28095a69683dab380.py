import time
import random

def construct():
    start = time.time()
    DEADLINE = 110.0  # safety margin under 120s
    
    # We need a Sidon set (all positive diffs distinct), containing 0,
    # covering differences 1..30, at most 32 elements, minimize max.
    
    def all_diffs(s):
        d = []
        n = len(s)
        for i in range(n):
            for j in range(i+1, n):
                d.append(s[j]-s[i])
        return d
    
    def is_sidon(s):
        d = all_diffs(s)
        return len(d) == len(set(d))
    
    def covers_1_30(s):
        d = set(all_diffs(s))
        return all(x in d for x in range(1,31))
    
    def valid(s):
        if len(s) > 32: return False
        if 0 not in s: return False
        if not is_sidon(s): return False
        if not covers_1_30(s): return False
        return True
    
    # Try to find a good solution via local search.
    # Start with a known-ish construction or random.
    
    best = None
    best_val = 10**9
    
    def try_update(s):
        nonlocal best, best_val
        if valid(s):
            v = max(s)
            if v < best_val:
                best_val = v
                best = sorted(s)
    
    # Initial: try greedy perfect-ish
    # Use a known small Golomb ruler seed then optimize.
    
    # Random restarts with local search
    random.seed(42)
    
    # Candidate generation: start from a set that covers 1..30
    # Use the "perfect" linear constraint to guide: sum (2k-32)*a_k = 465 for perfect.
    
    def random_initial():
        # Build a set that's likely Sidon and covers 1..30
        # Start with 0 and add elements to cover missing diffs
        s = [0]
        covered = set()
        # We'll add elements greedily to cover 1..30 while keeping Sidon
        missing = list(range(1,31))
        for d in missing:
            if d in covered:
                continue
            # try to add an element x such that x or x-d creates diff d
            found = False
            for _ in range(200):
                # pick random base
                base = random.choice(s)
                x = base + d
                if x < 0:
                    x = base - d
                if x in s:
                    continue
                trial = sorted(s + [x])
                if is_sidon(trial):
                    s = trial
                    # update covered
                    covered = set(all_diffs(s))
                    found = True
                    break
            if not found:
                # fallback: add far element
                x = 2*max(s) + d + 1
                while not is_sidon(sorted(s+[x])):
                    x += 1
                s = sorted(s+[x])
                covered = set(all_diffs(s))
        return s
    
    # Local search: reduce max element
    def local_search(s, iters=5000):
        s = sorted(s)
        cur_val = max(s)
        for it in range(iters):
            if time.time() > DEADLINE:
                break
            i = random.randrange(len(s))
            old = s[i]
            # try to move element i to a smaller value
            # new value in [0, cur_val]
            for _ in range(30):
                if time.time() > DEADLINE:
                    break
                new = random.randrange(0, cur_val+1)
                if new == old:
                    continue
                trial = sorted(s[:i] + [new] + s[i+1:])
                if len(set(trial)) < len(trial):
                    continue
                if is_sidon(trial) and covers_1_30(trial):
                    s = trial
                    cur_val = max(s)
                    break
        return s
    
    # Run several restarts
    for restart in range(200):
        if time.time() > DEADLINE:
            break
        s = random_initial()
        try_update(s)
        s = local_search(s, iters=3000)
        try_update(s)
    
    if best is None:
        # fallback to a guaranteed valid (the seed approach)
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
            else:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a
    
    return best
