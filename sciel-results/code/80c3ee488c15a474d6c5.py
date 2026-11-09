import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START


def _girth_cayley(n, S):
    """Exact girth of the Cayley graph of Z_n with symmetric generator set S,
    via non-backtracking BFS from every vertex (vectorized)."""
    m = len(S)
    if m == 0:
        return n + 1
    Sarr = np.array(S, dtype=np.int64)
    best = n + 1
    seen = np.full((n, m), -1, dtype=np.int64)
    for s in range(n):
        seen[:, :] = -1
        frontier = [(s, j) for j in range(m)]
        d = 1
        seen[s, j] = d
        while frontier and d < best:
            d += 1
            if d >= best:
                break
            # expand: for each (v, j), new vertices v + S, new direction i
            cur = np.array(frontier, dtype=np.int64).reshape(-1, 2)
            vs = cur[:, 0]
            js = cur[:, 1]
            # candidate new vertices for each new generator i
            # new vertex = v + S[i]
            cand = (vs[:, None] + Sarr[None, :]) % n
            dirs = np.broadcast_to(np.arange(m)[None, :], (len(vs), m))
            flat_v = cand.ravel()
            flat_d = dirs.ravel()
            # skip the backtracking edge (i == j)
            mask = (flat_d != js[:, None].ravel())
            flat_v = flat_v[mask]
            flat_d = flat_d[mask]
            # dedupe (v, i)
            uniq_v, first_idx = np.unique(flat_v, return_index=True)
            uniq_d = flat_d[first_idx]
            # check collisions with seen
            hit = seen[uniq_v, uniq_d] != -1
            if np.any(hit):
                cyc = d + 1
                if cyc < best:
                    best = cyc
                    if best < d + 1:
                        break
            # gather non-hit as new frontier
            keep = ~hit
            new_v = uniq_v[keep]
            new_d = uniq_d[keep]
            if len(new_v) == 0:
                break
            seen[new_v, new_d] = d
            frontier = list(zip(new_v.tolist(), new_d.tolist()))
    return best


def _cayley_edges(n, S):
    edges = []
    for v in range(n):
        for s in S:
            u = (v + s) % n
            if v < u:
                edges.append((v, u))
    return edges


def _cayley_degree_ok(n, S):
    # check every vertex has exactly 3 distinct neighbours
    Sset = set(S)
    for v in range(0, n, max(1, n // 2000)):
        nb = set((v + s) % n for s in Sset)
        if len(nb) != 3:
            return False
    # full check
    for v in range(n):
        nb = set((v + s) % n for s in Sset)
        if len(nb) != 3:
            return False
    return True


def _switch(n, S, i, j, a, b, rng):
    """Try a 2-switch: remove S[i], S[j]; add a, b. Keep 3-reg & simple."""
    newS = [x for x in S if x not in (S[i], S[j])] + [a, b]
    # must be symmetric and size 3
    if len(newS) != 3:
        return None
    Sset = set(newS)
    if len(Sset) != 3:
        return None
    # symmetry: S == -S
    for x in Sset:
        if (-x) % n not in Sset:
            return None
    # no 3-cycles: no x in Sset with 2x in Sset
    for x in Sset:
        if (2 * x) % n in Sset:
            return None
    # degree 3 everywhere (no collisions)
    for v in range(n):
        nb = set((v + x) % n for x in Sset)
        if len(nb) != 3:
            return None
    return newS


def construct(k, g):
    t0 = time.time()
    deadline = t0 + 140.0
    rng = random.Random(777)

    if k != 3:
        # generic fallback: cycle C_n is 2-regular; pad not possible -> return
        # a simple 3-regular on 2*(something). For safety return a known LUW-ish
        # but we only target k=3; return a trivial valid small graph if possible.
        # Build a 3-regular graph of girth >= g is hard generically; return cycle-based.
        # For k!=3 we still must return something valid-ish; use a 3-regular only if k==3.
        # Fallback: return a path? Must be k-regular. We'll just return empty-safe.
        # To avoid invalid, return a 3-regular only when k==3; else return minimal.
        # Provide a simple k-regular on 2k vertices (complete-ish) not guaranteed girth.
        n = 2 * k
        edges = []
        # build a k-regular: use circulant with generators 1..k-1? that's (k-1) reg.
        # Use complete graph K_k? not k-regular. Use 2 disjoint K_{k+1}? that's k-regular on 2(k+1)
        n = 2 * (k + 1)
        for grp in range(2):
            base = grp * (k + 1)
            for a in range(k + 1):
                for b in range(a + 1, k + 1):
                    edges.append((base + a, base + b))
        return n, edges

    # target k == 3
    # Moore bound for 3-regular girth 16 (even): 2*(2^{g/2 - 1}) = 2*2^7 = 512
    moore = 2 * (2 ** (g // 2 - 1))

    best = None
    best_n = None

    # Candidate n values (must be even for 3-regular, and n must allow a symmetric
    # 3-set with no 3-cycles: need n not divisible by 3, and even).
    cand_ns = []
    for n in range(moore, moore + 240, 2):
        if n % 2 != 0:
            continue
        if n % 3 == 0:
            continue
        cand_ns.append(n)

    for n in cand_ns:
        if time.time() > deadline - 5:
            break
        # enumerate symmetric 3-generator sets
        # generators a < b < c with -a,-b,-c in set => set is {a, -a, b} or {a, -a, -b}
        # i.e. one pair {x,-x} and one element y that equals -y? no, y must pair.
        # A symmetric 3-set in Z_n: must be {x, -x, y} where y = -y (so 2y=0) OR
        # {x,-x,y} with y paired -> need -y in set => -y = x or -x or y.
        #  -y = x => y = -x (dup). -y = -x => y = x (dup). -y = y => 2y = 0.
        # So symmetric 3-set = {x, -x, y} with 2y = 0 mod n. For even n, y = n/2.
        # Thus S = {x, -x, n/2}. Degree: neighbours v+x, v-x, v+n/2 distinct iff
        # x != 0, x != n/2, x != -x (2x!=0). 3-cycle if 2x in S or x + n/2 in S etc.
        half = n // 2
        for x in range(1, n):
            if x == 0 or x == half:
                continue
            if (2 * x) % n == 0:
                continue
            S = [x, (-x) % n, half]
            Sset = set(S)
            if len(Sset) != 3:
                continue
            # no 3-cycle: no element d in S with 2d in S
            bad = False
            for d in Sset:
                if (2 * d) % n in Sset:
                    bad = True
                    break
            if bad:
                continue
            # verify degree (should hold)
            ok = True
            for v in range(n):
                nb = set((v + d) % n for d in Sset)
                if len(nb) != 3:
                    ok = False
                    break
            if not ok:
                continue
            gi = _girth_cayley(n, S)
            if gi >= g:
                if best_n is None or n < best_n:
                    best_n = n
                    best = (n, _cayley_edges(n, S), S)
                break  # smallest n with valid S for this n; stop increasing n
        # local search: try to improve girth for this n if not found yet
        if best_n is not None and best_n <= n:
            break
        # attempt local search on a random valid S to push girth up
        if time.time() > deadline - 10:
            continue
        # find any valid S for this n to start from
        startS = None
        for x in range(1, n):
            if x == 0 or x == half:
                continue
            if (2 * x) % n == 0:
                continue
            S = [x, (-x) % n, half]
            Sset = set(S)
            if len(Sset) != 3:
                continue
            bad = any((2 * d) % n in Sset for d in Sset)
            if bad:
                continue
            ok = all(len(set((v + d) % n for d in Sset)) == 3 for v in range(n))
            if ok:
                startS = S
                break
        if startS is None:
            continue
        curS = startS
        curG = _girth_cayley(n, curS)
        for _ in range(3000):
            if time.time() > deadline - 5:
                break
            if curG >= g:
                if best_n is None or n < best_n:
                    best_n = n
                    best = (n, _cayley_edges(n, curS), curS)
                break
            # random switch
            i = rng.randrange(3)
            j = rng.randrange(3)
            if i == j:
                continue
            a = rng.randrange(1, n)
            b = rng.randrange(1, n)
            newS = _switch(n, curS, i, j, a, b, rng)
            if newS is None:
                continue
            newG = _girth_cayley(n, newS)
            if newG >= curG:
                curS = newS
                curG = newG
        if best_n is not None and best_n <= n:
            break

    if best is not None:
        n, edges, _ = best
        return n, edges

    # Fallback: return the smallest-n Cayley graph found even if girth < g
    # (better to return a valid 3-regular simple graph than nothing).
    # Use n = moore, S = {1, -1, n/2}
    n = moore
    if n % 2 != 0:
        n += 1
    half = n // 2
    S = [1, (-1) % n, half]
    edges = _cayley_edges(n, S)
    return n, edges

# EVOLVE-BLOCK-END
