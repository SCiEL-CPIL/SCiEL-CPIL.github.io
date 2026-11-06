"""Optimized Sidon set construction with brute-force search for minimal diameter."""
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def check(a):
        """Fast incremental Sidon check. Returns (ok, diff_set).
        ok is True iff a (sorted) is a Sidon set; diff_set is the set of positive diffs."""
        diffs = set()
        n = len(a)
        for i in range(n):
            ai = a[i]
            for j in range(i + 1, n):
                d = a[j] - ai
                if d in diffs:
                    return False, None
                diffs.add(d)
        return True, diffs

    # --- Try known difference sets first (cheap shortcut) ---
    # The (31, 6, 1) difference sets have diameter 30.
    # One such set is {0, 1, 5, 11, 25, 28} ? No, that one failed.
    # Let's rely on the brute force to find the correct one if it's not in the list.
    known_candidates = [
        [0, 1, 4, 10, 18, 23, 28], # (7,7,1) candidate, check if valid
        [0, 1, 3, 7, 12, 21, 28],
        [0, 1, 2, 7, 12, 22, 28],
        [0, 1, 3, 8, 12, 22, 26],
    ]
    for c in known_candidates:
        c = sorted(set(c))
        if len(c) < 1 or len(c) > 32:
            continue
        ok, cd = check(c)
        if ok and all(d in cd for d in range(1, 31)):
            return c

    # --- Brute force search for size 6 sets in 0..30 ---
    # A (31, 6, 1) difference set has 6 elements, max element 30, and covers 1..30.
    # Since min is 0, we choose 5 elements from 1..30.
    # Number of combinations: C(30, 5) = 142,506.
    # This is fast enough.
    target_size = 6
    max_val = 30
    base = [0]
    
    # We can also try size 7 with max 30? 
    # If a size 6 set exists, it's optimal.
    # If not, we might need larger sets, but diameter must be >= 30.
    # If we can't find size 6 in 0..30, we might need diameter > 30.
    
    # Optimization: 
    # We iterate through combinations of 5 numbers from 1..30.
    # We check if base + combo is Sidon and covers 1..30.
    
    found = None
    # To speed up, we can sort candidates? 
    # Just iterate.
    for combo in itertools.combinations(range(1, max_val + 1), target_size - 1):
        candidate = [0] + list(combo)
        ok, cd = check(candidate)
        if ok and all(d in cd for d in range(1, 31)):
            found = candidate
            break
        if time.time() > deadline - 10:
            break

    if found:
        return found

    # --- Fallback: Greedy construction if brute force fails or times out ---
    def greedy_construct():
        a = [0]
        covered = set()
        for d in range(1, 31):
            if d in covered:
                continue
            candidates = sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a))
            found = False
            for x in candidates:
                trial = a + [x]
                ok, nd = check(trial)
                if ok and d in nd:
                    a = sorted(trial)
                    covered = nd
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = a + [y, y + d]
                    ok, nd = check(trial)
                    if ok:
                        a = sorted(trial)
                        covered = nd
                        break
                    y += 1
        return a

    best = greedy_construct()
    best_max = max(best)

    # --- Local search to improve if needed ---
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            cur = list(best)
            max_val = max(cur)
            max_idx = cur.index(max_val)
            for new_val in range(1, max_val):
                if new_val in cur:
                    continue
                trial = cur[:max_idx] + [new_val] + cur[max_idx + 1:]
                trial = sorted(trial)
                if len(set(trial)) != len(trial):
                    continue
                ok, td = check(trial)
                if ok and all(d in td for d in range(1, 31)):
                    if max(trial) < best_max:
                        best = trial
                        best_max = max(trial)
                        improved = True
                        break
    return best
# EVOLVE-BLOCK-END
