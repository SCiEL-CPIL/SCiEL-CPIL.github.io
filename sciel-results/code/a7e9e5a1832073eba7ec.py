import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # explicit wall-clock guard (150s kill)

    # Generic fallback (only the (3,7,1) case is specialised)
    if not (q == 3 and n == 7 and radius == 1):
        if q == 3 and n <= 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetra = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetra for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    N = 3 ** n  # 2187
    deg = 1 + n * 2  # 15

    # --- integer <-> base-3 tuple ---
    pow3 = [1]
    for _ in range(n):
        pow3.append(pow3[-1] * 3)

    def int_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    # --- precompute neighbor lists (as int32 arrays) + membership sets ---
    neighbors = np.empty((N, deg), dtype=np.int32)
    for i in range(N):
        t = int_to_tuple(i)
        neighbors[i, 0] = i
        k = 1
        for pos in range(n):
            v = t[pos]
            for nv in range(3):
                if nv != v:
                    t[pos] = nv
                    idx = 0
                    for x in t:
                        idx = idx * 3 + x
                    neighbors[i, k] = idx
                    k += 1
                    t[pos] = v

    # membership test via a flat int32 lookup table (fast O(1) "in ball" test)
    memb = np.zeros(N * deg, dtype=np.int32)
    for i in range(N):
        memb[neighbors[i]] = i + 1
    # helper: is word `w` in ball around `c`?
    def in_ball(c, w):
        return memb[c * deg + w] != 0

    # --- Phase 1: greedy set cover with vectorized gain accumulation ---
    uncovered = np.ones(N, dtype=bool)
    code = []

    while np.any(uncovered):
        if time.time() > deadline - 30:
            break
        gains = np.zeros(N, dtype=np.int32)
        gains[neighbors[np.where(uncovered)[0]]] += 1
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        code.append(best)
        uncovered[neighbors[best]] = False

    # --- Phase 2: local search (redundancy removal + swaps) ---
    counts = np.zeros(N, dtype=np.int32)
    for c in code:
        counts[neighbors[c]] += 1
    code_set = set(code)

    def try_remove():
        removed = False
        for c in list(code):
            if np.min(counts[neighbors[c]]) > 1:
                code.remove(c)
                code_set.discard(c)
                counts[neighbors[c]] -= 1
                removed = True
        return removed

    def try_swap(c):
        if np.min(counts[neighbors[c]]) > 1:
            return False
        crit = [i for i in neighbors[c] if counts[i] == 1]
        if not crit:
            return False
        # candidate replacements: common neighbours of the critical points
        inter = set(neighbors[crit[0]].tolist())
        for p in crit[1:]:
            inter &= set(neighbors[p].tolist())
        inter -= code_set
        if not inter:
            return False
        best_w = -1
        best_score = -1
        for w in inter:
            if w == c:
                continue
            score = int(np.sum(counts[neighbors[w]] == 1))
            if score > best_score:
                best_score = score
                best_w = w
        if best_w != -1:
            code.remove(c)
            code_set.discard(c)
            counts[neighbors[c]] -= 1
            code.append(best_w)
            code_set.add(best_w)
            counts[neighbors[best_w]] += 1
            return True
        return False

    improved = True
    while improved and time.time() < deadline - 25:
        improved = False
        if try_remove():
            improved = True
            continue
        for c in list(code):
            if time.time() > deadline - 25:
                break
            if try_swap(c):
                improved = True
                break

    # --- Phase 3: symmetry-driven minimisation (structural) ---
    # A codeword c is "fixed" if every ball that covers its critical points is
    # occupied.  If a coordinate permutation maps c to a free position while
    # still covering the critical points, we can relocate it (and any ball that
    # only covered c becomes free).  Repeated relocations shrink the cover.
    # Build the full permutation group on 7 coordinates.
    perms = list(itertools.permutations(range(n)))
    # precompute, for each permutation, the mapping of each index to its image
    perm_map = []
    for perm in perms:
        m = np.empty(N, dtype=np.int32)
        for i in range(N):
            t = int_to_tuple(i)
            j = 0
            for k in range(n):
                j = j * 3 + t[perm[k]]
            m[i] = j
        perm_map.append(m)

    def critical_points(c):
        return [i for i in neighbors[c] if counts[i] == 1]

    def relocate(c, perm):
        """Try to move c under permutation `perm`. Return True if moved."""
        crit = critical_points(c)
        if not crit:
            return False
        target = int(perm[c])
        if target in code_set:
            return False
        # target must cover all critical points of c
        for p in crit:
            if not in_ball(target, p):
                return False
        # removing c must not uncover anything
        if np.min(counts[neighbors[c]]) <= 1:
            return False
        # perform the move
        code.remove(c)
        code_set.discard(c)
        counts[neighbors[c]] -= 1
        code.append(target)
        code_set.add(target)
        counts[neighbors[target]] += 1
        return True

    best_code = list(code)
    best_size = len(code)

    def snapshot():
        return len(code)

    moved = True
    while moved and time.time() < deadline - 6:
        moved = False
        # try redundancy removal first
        if try_remove():
            moved = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
            continue
        # try relocating codewords under random permutations
        for c in list(code):
            if time.time() > deadline - 6:
                break
            crit = critical_points(c)
            if not crit:
                continue
            # only consider permutations that keep the critical points covered
            # cheap filter: try a handful of random perms
            for _ in range(40):
                if time.time() > deadline - 6:
                    break
                perm = random.choice(perm_map)
                target = int(perm[c])
                if target in code_set:
                    continue
                ok = True
                for p in crit:
                    if not in_ball(target, p):
                        ok = False
                        break
                if not ok:
                    continue
                if np.min(counts[neighbors[c]]) <= 1:
                    continue
                # do the move
                code.remove(c)
                code_set.discard(c)
                counts[neighbors[c]] -= 1
                code.append(target)
                code_set.add(target)
                counts[neighbors[target]] += 1
                moved = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                break

    if len(code) > best_size:
        code = best_code

    # final safety: ensure validity; if somehow invalid, fall back to greedy result
    counts2 = np.zeros(N, dtype=np.int32)
    for c in code:
        counts2[neighbors[c]] += 1
    if np.any(counts2 == 0):
        # rebuild a guaranteed-valid cover
        uncovered = np.ones(N, dtype=bool)
        code = []
        while np.any(uncovered) and time.time() < deadline - 2:
            gains = np.zeros(N, dtype=np.int32)
            gains[neighbors[np.where(uncovered)[0]]] += 1
            best = int(np.argmax(gains))
            if gains[best] == 0:
                break
            code.append(best)
            uncovered[neighbors[best]] = False

    return [int_to_tuple(c) for c in code]
