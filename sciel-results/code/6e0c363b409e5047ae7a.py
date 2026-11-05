"""Ambient integer sets with small largest Sidon subset.
We want a 12-element set in [0,128] whose largest 'strong Sidon' subset
(all unordered pair sums including diagonals unique) is as small as possible."""
import random
import time
import math
from itertools import combinations


def max_sidon_size(S):
    """Exact largest k such that some k-subset of S has all unordered pair sums
    (including diagonals) unique. Returns that k."""
    S = sorted(S)
    n = len(S)
    if n <= 1:
        return n

    # Precompute for each element i: list of sum-IDs it uses (with elements j >= i)
    # and a bitmask of all sums it touches
    allsums = set()
    for i in range(n):
        for j in range(i, n):
            allsums.add(S[i] + S[j])
    allsums = sorted(allsums)
    sum_to_id = {s: k for k, s in enumerate(allsums)}
    m = len(allsums)

    # For each i, compute the list of sum-ids and a bitmask
    sums_ids_by_i = []
    masks_by_i = []
    for i in range(n):
        ids = []
        mask = 0
        for j in range(i, n):
            sid = sum_to_id[S[i] + S[j]]
            ids.append(sid)
            mask |= (1 << sid)
        sums_ids_by_i.append(ids)
        masks_by_i.append(mask)

    # Order elements by "conflict degree" - how many other elements share a sum
    # Elements with more conflicts should be tried first (they fail fast)
    conflict_degree = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            # Check if S[i]+S[j] collides with any other pair
            for k in range(n):
                if k == i or k == j:
                    continue
                for l in range(k, n):
                    if l == i or l == j:
                        continue
                    if S[i] + S[j] == S[k] + S[l]:
                        conflict_degree[i] += 1
                        conflict_degree[j] += 1
                        break

    # Sort indices by conflict degree (descending) for better pruning
    # But we need to maintain the "start" invariant for subset enumeration
    # Instead, we'll use a different strategy: try to build up from small sets

    best = 1
    used_mask = 0

    # DFS: pick elements one by one, maintaining that all pair sums are unique
    # Key insight: when we add element i, it conflicts with used_mask if
    # any of its sums (with previously chosen elements) are already used
    # But we also need to track sums within the chosen set

    # Better approach: state = (chosen_mask, used_sum_mask)
    # But chosen_mask can be up to 2^12 = 4096, and used_sum_mask is large
    # Let's use memoization on (chosen_set_as_tuple, )

    # Actually, let's use a cleaner recursive approach with bitmasks
    # State: which elements are chosen (bitmask), and the set of used sums (bitmask)
    # But the used sums depend on which elements are chosen

    # Simpler: just do DFS with early termination
    # Order elements to try first: those with highest conflict degree

    # For the DFS, we process elements in sorted order but try to add them
    # The key optimization: if adding element i would create a sum collision
    # with any already-chosen element, skip it

    chosen = [False] * n
    used_sums = set()

    def dfs(idx, size):
        nonlocal best
        if size > best:
            best = size
        if best == n:
            return
        if idx == n:
            return
        # Prune: even if we take all remaining, can't beat best
        if size + (n - idx) <= best:
            return

        # Option 1: skip element idx
        dfs(idx + 1, size)

        # Option 2: try to add element idx
        # Check if S[idx] + S[j] for all chosen j (and S[idx]+S[idx]) are new
        ok = True
        new_sums = []
        # Diagonal sum
        diag = S[idx] + S[idx]
        if diag in used_sums:
            ok = False
        else:
            new_sums.append(diag)
        # Sums with previously chosen elements
        if ok:
            for j in range(n):
                if chosen[j]:
                    s = S[idx] + S[j]
                    if s in used_sums:
                        ok = False
                        break
                    new_sums.append(s)

        if ok:
            chosen[idx] = True
            for s in new_sums:
                used_sums.add(s)
            dfs(idx + 1, size + 1)
            for s in new_sums:
                used_sums.discard(s)
            chosen[idx] = False

    dfs(0, 0)
    return best


def _valid(c):
    if len(c) != 12 or len(set(c)) != 12:
        return False
    return all(0 <= x <= 128 for x in c)


def construct():
    start = time.time()
    deadline = start + 87.0  # safety margin under 90s

    best_set = None
    best_val = 99
    seen = set()

    def consider(c):
        nonlocal best_set, best_val
        if time.time() > deadline:
            return None
        key = tuple(sorted(c))
        if key in seen:
            return None
        seen.add(key)
        if not _valid(c):
            return None
        v = max_sidon_size(c)
        if v < best_val:
            best_val = v
            best_set = list(c)
        return v

    rng = random.Random(42)

    # Phase 1: Try consecutive sets (known to have small Sidon subsets)
    for lo in range(0, 117):
        if time.time() > deadline:
            break
        c = list(range(lo, lo + 12))
        consider(c)

    # Phase 2: Structured sets with many sum collisions
    # Two interleaved APs
    for d in range(1, 8):
        if time.time() > deadline:
            break
        for a in range(0, 100):
            for gap in range(1, 15):
                b = a + gap
                if b + 5 * d > 128:
                    continue
                c1 = [a + i * d for i in range(6)]
                c2 = [b + i * d for i in range(6)]
                c = sorted(set(c1 + c2))
                if len(c) == 12 and _valid(c):
                    consider(c)

    # Phase 3: Random sets in small windows
    for trial in range(3000):
        if time.time() > deadline:
            break
        lo = rng.randint(0, 90)
        width = rng.randint(12, 25)
        hi = min(lo + width, 129)
        if hi - lo < 12:
            continue
        c = sorted(rng.sample(range(lo, hi), 12))
        consider(c)

    # Phase 4: Local search - move elements to create more collisions
    if best_set is not None:
        cur = list(best_set)
        cur_val = best_val
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                c2 = list(cur)
                pos = rng.randrange(12)
                delta = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
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
                    improved = True
                elif v == cur_val and rng.random() < 0.2:
                    cur = c2
        consider(cur)

    # Phase 5: If we have time, try more structured patterns
    # Sets with repeated differences
    if time.time() < deadline - 10:
        for base in range(0, 50):
            if time.time() > deadline - 5:
                break
            for d1 in range(1, 10):
                for d2 in range(1, 10):
                    c = []
                    # 6 elements from AP with step d1, 6 from AP with step d2
                    c1 = [base + i * d1 for i in range(6)]
                    c2 = [base + d2 + i * d2 for i in range(6)]
                    c = sorted(set(c1 + c2))
                    if len(c) == 12 and _valid(c):
                        consider(c)

    # If nothing found, return a default
    if best_set is None:
        best_set = list(range(12))

    return best_set
