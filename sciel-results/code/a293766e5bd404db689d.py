"""Kissing number in dimension 11 via D11 lattice maximum independent set."""
import time
import numpy as np
import itertools

D = 11


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # ------------------------------------------------------------------
    # Generate all integer vectors with squared norm exactly 4
    #   Type A: one coordinate +/-2  -> 2*D = 22
    #   Type B: four coordinates +/-1 -> C(D,4)*2^4 = 5280
    # ------------------------------------------------------------------
    vecs = []
    # Type A
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vecs.append(v)
    # Type B
    for coords in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for c, s in zip(coords, signs):
                v[c] = s
            vecs.append(v)

    n = len(vecs)
    V = np.array(vecs, dtype=np.int64)

    # ------------------------------------------------------------------
    # Conflict graph: edge (i,j) if dot(i,j) > 2  (i.e. |v_i - v_j|^2 < 4)
    # We need an independent set: no two selected vectors conflict.
    # ------------------------------------------------------------------
    dot = V @ V.T
    conflict = (dot > 2)
    np.fill_diagonal(conflict, False)

    # Bitset adjacency for fast neighbor queries
    # Use Python ints as arbitrary-precision bitsets
    adj_bits = [0] * n
    for i in range(n):
        rows = np.where(conflict[i])[0]
        b = 0
        for r in rows:
            b |= (1 << r)
        adj_bits[i] = b

    degrees = np.array([bin(b).count('1') for b in adj_bits], dtype=np.int32)

    # ------------------------------------------------------------------
    # Maximum independent set via branch-and-bound on the conflict graph.
    # The independent set in G = clique in complement. We work directly
    # on G: pick a vertex, remove it and its neighbors, recurse.
    # ------------------------------------------------------------------
    # Use a greedy coloring upper bound for pruning.
    # We represent the candidate vertex set as a bitset (Python int).

    def popcount(x):
        return bin(x).count('1')

    def highest_bit(x):
        return x.bit_length() - 1

    def lowest_bit(x):
        return (x & (-x)).bit_length() - 1

    def next_bit(x):
        # return index of least significant set bit
        return (x & (-x)).bit_length() - 1

    def iter_bits(x):
        while x:
            lsb = x & (-x)
            yield lsb.bit_length() - 1
            x ^= lsb

    # Greedy coloring of a candidate bitset for upper bound.
    # Colors vertices in order of (heuristic) and assigns smallest available.
    def greedy_color_ub(cand):
        # Order: pick highest degree first for tighter bound? Use simple order.
        # We'll process in descending degree order.
        idxs = list(iter_bits(cand))
        # sort by degree descending
        idxs.sort(key=lambda i: -degrees[i])
        color_of = {}
        for i in idxs:
            # find smallest color not used by colored neighbors in cand
            used = set()
            nb = adj_bits[i] & cand
            for j in iter_bits(nb):
                if j in color_of:
                    used.add(color_of[j])
            c = 0
            while c in used:
                c += 1
            color_of[i] = c
        return max(color_of.values()) + 1 if color_of else 0

    best_size = 0
    best_set = 0  # bitset of chosen independent set

    full = (1 << n) - 1

    def solve(cand):
        nonlocal best_size, best_set
        if cand == 0:
            return
        sz = popcount(cand)
        if sz <= best_size:
            return
        # Upper bound via greedy coloring
        ub = greedy_color_ub(cand)
        if ub <= best_size:
            return
        if sz == best_size + ub - ub:  # not useful
            pass
        # If the whole set is independent, take it
        if not any(adj_bits[i] & cand for i in iter_bits(cand)):
            if sz > best_size:
                best_size = sz
                best_set = cand
            return
        # Choose pivot vertex: one with max degree within cand
        # Branch: include or exclude pivot.
        # We try include first (greedy bias) then exclude.
        # Pick pivot as vertex with max (degree within cand)
        pivot = -1
        best_deg = -1
        for i in iter_bits(cand):
            d = popcount(adj_bits[i] & cand)
            if d > best_deg:
                best_deg = d
                pivot = i
        pb = 1 << pivot
        # Include pivot
        cand_include = cand & ~(pb | adj_bits[pivot])
        # Exclude pivot
        cand_exclude = cand & ~pb
        # Explore include first if it can beat best
        # Recursive include
        # To keep it tractable, do a depth-limited DFS with time check.
        # We'll implement iterative stack to avoid recursion depth issues.
        stack = [cand_include, cand_exclude]
        while stack:
            if time.time() > deadline - 8:
                return
            c = stack.pop()
            if c == 0:
                continue
            szc = popcount(c)
            if szc <= best_size:
                continue
            ubc = greedy_color_ub(c)
            if ubc <= best_size:
                continue
            if not any(adj_bits[i] & c for i in iter_bits(c)):
                if szc > best_size:
                    best_size = szc
                    best_set = c
                continue
            # branch
            pivot2 = -1
            bd2 = -1
            for i in iter_bits(c):
                d2 = popcount(adj_bits[i] & c)
                if d2 > bd2:
                    bd2 = d2
                    pivot2 = i
            p2b = 1 << pivot2
            stack.append(c & ~(p2b | adj_bits[pivot2]))
            stack.append(c & ~p2b)

    # Seed the search with a greedy independent set first
    def greedy_seed():
        chosen = 0
        remaining = full
        # pick low-degree vertices first
        order = np.argsort(degrees)
        for i in order:
            if time.time() > deadline - 10:
                break
            if (1 << i) & remaining:
                chosen |= (1 << i)
                remaining &= ~(1 << i) & ~adj_bits[i]
        return chosen

    gset = greedy_seed()
    best_size = popcount(gset)
    best_set = gset

    solve(full)

    # ------------------------------------------------------------------
    # Fallback: if B&B found nothing better, run local-search polish.
    # ------------------------------------------------------------------
    if time.time() < deadline - 10 and best_size < 480:
        # local search: try to add any non-conflicting vertex
        chosen_bool = np.zeros(n, dtype=bool)
        for i in iter_bits(best_set):
            chosen_bool[i] = True
        # repeatedly add a vertex with zero conflicts among chosen
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            added = False
            for i in range(n):
                if chosen_bool[i]:
                    continue
                if not (conflict[i] & chosen_bool).any():
                    chosen_bool[i] = True
                    added = True
                    break
            if not added:
                break
        best_set = 0
        for i in np.where(chosen_bool)[0]:
            best_set |= (1 << i)
        best_size = popcount(best_set)

    # ------------------------------------------------------------------
    # Extract final index list
    # ------------------------------------------------------------------
    idxs = list(iter_bits(best_set))
    if not idxs:
        # Fallback: return a small valid set (type A with one sign per axis)
        idxs = list(range(D))

    return V[np.array(idxs, dtype=np.int64)]
