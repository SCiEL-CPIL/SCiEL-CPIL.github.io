import itertools
import time

def _square(c):
    n = len(c)
    out = [0] * (2 * n - 1)
    for i in range(n):
        a = c[i]
        if a:
            for j in range(i, n):
                v = a * c[j]
                out[i + j] += v
                if i != j:
                    out[i + j] += v
    return out

def _nz(x):
    return sum(1 for v in x if v != 0)

def _collision_count(pos, n=12):
    # number of (unordered i<j) pairs sharing a sum
    s = {}
    for i in range(n):
        for j in range(i + 1, n):
            s[pos[i] + pos[j]] = s.get(pos[i] + pos[j], 0) + 1
    return sum(v * (v - 1) // 2 for v in s.values())

def _candidate_sums(pos, n=12):
    s = set()
    for i in range(n):
        for j in range(i, n):
            s.add(pos[i] + pos[j])
    return s

def _solve_values(pos, n=12, rng=None):
    # Build A (m x n) rows = candidate sums, then find a 0/1 value vector
    # (entries in {-1,1}) maximizing number of zero candidate-sum rows.
    cand = _candidate_sums(pos, n)
    rows = []
    for s in cand:
        row = [0] * n
        for i in range(n):
            for j in range(i, n):
                if pos[i] + pos[j] == s:
                    if i == j:
                        row[i] += 1
                    else:
                        row[i] += 1
                        row[j] += 1
        rows.append(row)
    m = len(rows)
    best = None
    best_zero = -1
    if rng is not None:
        for _ in range(40):
            v = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
            z = 0
            for row in rows:
                t = 0
                for k in range(n):
                    if row[k]:
                        t += row[k] * v[k]
                if t == 0:
                    z += 1
            if z > best_zero:
                best_zero = z
                best = list(v)
    # local search: flip single values
    if best is None:
        best = [1] * n
    improved = True
    while improved:
        improved = False
        for k in range(n):
            for sgn in (-1, 1):
                if best[k] == sgn:
                    continue
                nb = list(best)
                nb[k] = sgn
                z = 0
                for row in rows:
                    t = 0
                    for i in range(n):
                        if row[i]:
                            t += row[i] * nb[i]
                    if t == 0:
                        z += 1
                if z > best_zero:
                    best_zero = z
                    best = nb
                    improved = True
    # scale: try small integer magnitudes to keep all-zero rows zero (they stay zero)
    # and try to keep square small; choose scale that minimizes nonzero of full square
    best_scale = 1
    best_nz = None
    for sc in (1, 2, 3, 4, 5, 8):
        c = [0] * 41
        for i in range(n):
            c[pos[i]] = best[i] * sc
        sq = _square(c)
        nz = _nz(sq)
        if best_nz is None or nz < best_nz:
            best_nz = nz
            best_scale = sc
    c = [0] * 41
    for i in range(n):
        c[pos[i]] = best[i] * best_scale
    return c, best_nz

def construct():
    start = time.time()
    deadline = start + 80.0
    import random
    rng = random.Random(12345)
    best_c = [1] * 12
    best_score = _nz(_square(best_c))
    best_pos = list(range(12))

    # Phase 1: search for positions with many collisions (fewer candidate sums)
    # Start from a few heuristic structures, then random perturbation.
    candidates = [list(range(12))]
    # arithmetic-ish: 0,1,2,3,4,5,6,7,8,9,10,11 already
    # add some structured: symmetric around center
    candidates.append([0,1,2,3,4,5,6,7,8,9,10,11])
    # clustered
    candidates.append([0,1,2,3,4,5,6,7,8,9,10,11])
    # random seeds
    for _ in range(20):
        pos = sorted(rng.sample(range(41), 12))
        candidates.append(pos)

    # local search on positions: swap two positions, accept if collision count improves
    def pos_coll(pos):
        return _collision_count(pos)

    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(12):
            for j in range(i + 1, 12):
                if time.time() > deadline - 10.0:
                    break
                nb = list(candidates[0])
                nb[i], nb[j] = nb[j], nb[i]
                nb.sort()
                if pos_coll(nb) > pos_coll(candidates[0]):
                    candidates[0] = nb
                    improved = True
            if time.time() > deadline - 10.0:
                break
        # also try random restarts if stuck
        if not improved:
            for _ in range(5):
                pos = sorted(rng.sample(range(41), 12))
                if pos_coll(pos) > pos_coll(candidates[0]):
                    candidates[0] = pos
                    improved = True

    best_pos = candidates[0]

    # Phase 2: for the best position, solve values; also try a few nearby positions
    positions_to_try = [best_pos]
    # neighbors: swap pairs
    for i in range(12):
        for j in range(i + 1, 12):
            nb = list(best_pos)
            nb[i], nb[j] = nb[j], nb[i]
            nb.sort()
            positions_to_try.append(nb)
    # a few random
    for _ in range(10):
        positions_to_try.append(sorted(rng.sample(range(41), 12)))

    for pos in positions_to_try:
        if time.time() > deadline - 3.0:
            break
        c, nz = _solve_values(pos, 12, rng)
        if nz < best_score:
            best_score = nz
            best_c = c
            best_pos = pos

    # Final local polish on values (flip signs / try magnitudes) for best_c
    def polish(c):
        best = list(c)
        best_nz = _nz(_square(best))
        # flip a single entry sign
        for k in range(41):
            if best[k] == 0:
                continue
            nb = list(best)
            nb[k] = -nb[k]
            nz = _nz(_square(nb))
            if nz < best_nz:
                best = nb
                best_nz = nz
        # try small magnitude changes
        for k in range(41):
            if best[k] == 0:
                continue
            for sc in range(1, 9):
                nb = list(best)
                sgn = 1 if best[k] > 0 else -1
                nb[k] = sgn * sc
                nz = _nz(_square(nb))
                if nz < best_nz:
                    best = nb
                    best_nz = nz
        return best, best_nz

    if time.time() < deadline - 2.0:
        best_c, best_score = polish(best_c)

    return best_c
