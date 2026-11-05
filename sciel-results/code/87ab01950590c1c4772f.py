"""Ambient integer sets with small largest Sidon subset.
We want a 12-element set in [0,128] whose largest 'strong Sidon' subset
(all unordered pair sums including diagonals unique) is as small as possible."""
import random
import time
import math


def max_sidon_size(S):
    """Exact largest k such that some k-subset of S has all unordered pair sums
    (including diagonals) unique. Returns that k."""
    S = list(S)
    n = len(S)
    if n <= 1:
        return n

    # Precompute pair sums for each i: sums of S[i] + S[j] for j >= i
    sums_by_i = [
        [S[i] + S[j] for j in range(i, n)]
        for i in range(n)
    ]

    # Map each distinct sum to an ID
    allsums = set()
    for lst in sums_by_i:
        allsums.update(lst)
    sum_to_id = {s: k for k, s in enumerate(allsums)}
    m = len(allsums)

    best = 1
    cnt = [0] * m

    # Use a bitmask-free approach but with aggressive pruning.
    # Order: try elements that create the most conflicts first (to fail fast)
    # Actually, for maximizing, we want to try adding elements.
    # Better: use a different search strategy.

    # Let's use a different approach: try to find a large Sidon subset.
    # We'll do DFS with ordering by "conflict potential".

    # Precompute for each element i, how many sums it shares with other elements
    # This helps with ordering.
    # For now, just do straightforward DFS with pruning.

    def dfs(start, size):
        nonlocal best
        if size > best:
            best = size
        if size + (n - start) <= best:
            return
        if best == n:
            return
        for i in range(start, n):
            ok = True
            touched = []
            for s in sums_by_i[i]:
                sid = sum_to_id[s]
                if cnt[sid] >= 1:
                    ok = False
                    break
                cnt[sid] += 1
                touched.append(sid)
            if ok:
                dfs(i + 1, size + 1)
            for sid in touched:
                cnt[sid] -= 1

    dfs(0, 0)
    return best


def _valid(c):
    if len(c) != 12 or len(set(c)) != 12:
        return False
    return all(0 <= x <= 128 for x in c)


def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s

    best_set = None
    best_val = 99
    seen = set()

    def consider(c):
        nonlocal best_set, best_val
        key = tuple(sorted(c))
        if key in seen:
            return
        seen.add(key)
        if not _valid(c):
            return
        v = max_sidon_size(c)
        if v < best_val:
            best_val = v
            best_set = list(c)
        return v

    # Strategy: We want to MINIMIZE the largest Sidon subset.
    # A set with many additive collisions (many pairs sharing sums) will have small Sidon subsets.
    # Arithmetic progressions are bad for Sidon (large Sidon subsets).
    # We want sets with MANY collisions in pair sums.
    #
    # Idea: use sets that are "dense" in a small range, or have structure that creates many sum collisions.
    # For example, a set like {0,1,2,3,4,5,6,7,8,9,10,11} has many collisions.
    # Let's test small ranges first.

    # Try all sets in small ranges
    for lo in range(0, 120):
        if time.time() > deadline:
            break
        hi = lo + 12
        if hi > 129:
            break
        # Try the consecutive set
        c = list(range(lo, lo + 12))
        consider(c)

    # Try sets with small gaps but not all consecutive
    rng = random.Random(42)
    for trial in range(5000):
        if time.time() > deadline:
            break
        # Random set in a small window
        lo = rng.randint(0, 100)
        width = rng.randint(12, 30)
        hi = min(lo + width, 129)
        if hi - lo < 12:
            continue
        c = sorted(rng.sample(range(lo, hi), 12))
        consider(c)

    # Try structured sets with many collisions
    # Sets like {a, a+d, a+2d, ..., a+5d, b, b+d, ..., b+5d} where b-a is small
    for d in range(1, 10):
        if time.time() > deadline:
            break
        for a in range(0, 120 - d * 11):
            for gap in range(1, 20):
                b = a + gap
                if b + 5 * d > 128:
                    continue
                c1 = [a + i * d for i in range(6)]
                c2 = [b + i * d for i in range(6)]
                c = sorted(set(c1 + c2))
                if len(c) == 12 and _valid(c):
                    consider(c)

    # Local search from best found
    if best_set is not None:
        cur = list(best_set)
        cur_val = best_val
        for _ in range(200):
            if time.time() > deadline:
                break
            c2 = list(cur)
            pos = rng.randrange(12)
            # Try to move an element to create more collisions
            # Pick a nearby value
            delta = rng.choice([-3, -2, -1, 1, 2, 3])
            nv = c2[pos] + delta
            if nv < 0 or nv > 128:
                continue
            if nv in c2:
                continue
            c2[pos] = nv
            c2.sort()
            v = max_sidon_size(c2)
            if v < cur_val:
                cur = c2
                cur_val = v
            elif v == cur_val and rng.random() < 0.3:
                cur = c2
        consider(cur)

    # If nothing found, return a default
    if best_set is None:
        best_set = list(range(12))

    return best_set
