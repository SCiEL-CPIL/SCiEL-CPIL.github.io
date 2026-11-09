import time
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    t0 = time.time()
    DEADLINE = t0 + 80.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        a, b = (u, v) if u < v else (v, u)
        adj[a].add(b)
        adj[b].add(a)
        edge_set.add((a, b))
    total_edges = len(edge_set)

    def me(u, v):
        return (u, v) if u < v else (v, u)

    # ---- NumPy bitmask representation of the remaining edge set ----
    # B[i,j] = 1  <=>  edge {i,j} is currently remaining
    B = np.zeros((n, n), dtype=np.int64)
    for (a, b) in edge_set:
        B[a, b] = 1
        B[b, a] = 1

    def remaining_edges_from_B(Bm):
        out = set()
        for i in range(n):
            js = np.nonzero(Bm[i])[0]
            for j in js:
                if j > i:
                    out.add((int(i), int(j)))
        return out

    def cross_mask(Bm, L, R):
        """Edges in Bm between L and R, as a bool matrix (symmetric)."""
        m = np.zeros((n, n), dtype=bool)
        for l in L:
            for r in R:
                m[l, r] = Bm[l, r] == 1
                m[r, l] = Bm[l, r] == 1
        return m

    def expand_biclique(L, R, Bm):
        """Maximally expand (L,R) within the current remaining edges."""
        L = set(L); R = set(R)
        if not L or not R:
            return L, R
        changed = True
        while changed:
            changed = False
            # expand L
            cand = list(range(n))
            newL = []
            for w in cand:
                if w in L or w in R:
                    continue
                ok = True
                for r in R:
                    if Bm[w, r] != 1:
                        ok = False
                        break
                if ok:
                    newL.append(w)
            if newL:
                for w in newL:
                    L.add(w)
                changed = True
            # expand R
            newR = []
            for w in cand:
                if w in L or w in R:
                    continue
                ok = True
                for l in L:
                    if Bm[w, l] != 1:
                        ok = False
                        break
                if ok:
                    newR.append(w)
            if newR:
                for w in newR:
                    R.add(w)
                changed = True
        return L, R

    def find_max_biclique(Bm, strategy, deadline):
        best = None
        best_size = 0
        if strategy == 'expand':
            for (u, v) in remaining_edges_from_B(Bm):
                if time.time() > deadline:
                    break
                L, R = expand_biclique({u}, {v}, Bm)
                sz = len(L) * len(R)
                if sz > best_size:
                    best_size = sz
                    best = (L, R)
        elif strategy == 'common':
            for u in range(n):
                if time.time() > deadline:
                    break
                nb = set()
                for v in range(n):
                    if v != u and Bm[u, v] == 1:
                        nb.add(v)
                if not nb:
                    continue
                L, R = expand_biclique({u}, set(nb), Bm)
                sz = len(L) * len(R)
                if sz > best_size:
                    best_size = sz
                    best = (L, R)
        elif strategy == 'degree':
            deg = [0] * n
            for i in range(n):
                deg[i] = int(np.sum(Bm[i]))
            order = sorted(range(n), key=lambda v: -deg[v])
            for u in order[:25]:
                if time.time() > deadline:
                    break
                nb = set()
                for v in range(n):
                    if v != u and Bm[u, v] == 1:
                        nb.add(v)
                if not nb:
                    continue
                L, R = expand_biclique({u}, set(nb), Bm)
                sz = len(L) * len(R)
                if sz > best_size:
                    best_size = sz
                    best = (L, R)
        return best

    def greedy_partition(Bm, strategies, deadline):
        bicliques = []
        Bm = Bm.copy()
        while True:
            if time.time() > deadline:
                break
            if not np.any(Bm):
                break
            best = None
            best_size = 0
            for s in strategies:
                bc = find_max_biclique(Bm, s, deadline)
                if bc is not None:
                    L, R = bc
                    sz = len(L) * len(R)
                    if sz > best_size:
                        best_size = sz
                        best = bc
            if best is None:
                # fallback: single edges
                for (u, v) in remaining_edges_from_B(Bm):
                    bicliques.append((frozenset([u]), frozenset([v])))
                Bm = np.zeros((n, n), dtype=np.int64)
                break
            L, R = best
            bicliques.append((frozenset(L), frozenset(R)))
            for l in L:
                for r in R:
                    if l != r:
                        Bm[l, r] = 0
                        Bm[r, l] = 0
        # leftover -> single edges
        for (u, v) in remaining_edges_from_B(Bm):
            bicliques.append((frozenset([u]), frozenset([v])))
        return bicliques

    def graph_valid_biclique(L, R):
        if not L or not R:
            return False
        if L & R:
            return False
        for l in L:
            for r in R:
                if l != r and me(l, r) not in edge_set:
                    return False
        return True

    def try_merge(bicliques, edge_set, deadline):
        nb = len(bicliques)
        for i in range(nb):
            for j in range(i + 1, nb):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                cands = []
                # union both
                a = L1 | L2; b = R1 | R2
                if a and b and not (a & b):
                    cands.append((a, b))
                # cross
                a = L1 | R2; b = R1 | L2
                if a and b and not (a & b):
                    cands.append((a, b))
                # subset absorbs
                if L2 <= L1:
                    b = R1 | R2
                    if b and not (L1 & b):
                        cands.append((L1, b))
                if L1 <= L2:
                    b = R1 | R2
                    if b and not (L2 & b):
                        cands.append((L2, b))
                if R2 <= R1:
                    a = L1 | L2
                    if a and not (a & R1):
                        cands.append((a, R1))
                if R1 <= R2:
                    a = L1 | L2
                    if a and not (a & R2):
                        cands.append((a, R2))
                for a, b in cands:
                    if graph_valid_biclique(a, b):
                        res = list(bicliques)
                        res.pop(j)
                        res.pop(i)
                        res.append((frozenset(a), frozenset(b)))
                        return res
        return None

    def try_absorb(bicliques, edge_set, deadline):
        """Try to fold a small biclique's vertices into a larger one."""
        nb = len(bicliques)
        for j in range(nb):
            for i in range(nb):
                if i == j:
                    continue
                Li, Ri = bicliques[i]
                Lj, Rj = bicliques[j]
                # try moving a vertex of Lj into Li or Ri
                for v in list(Lj):
                    # v -> Li
                    a = Li | {v}; b = Ri
                    if a and b and not (a & b) and graph_valid_biclique(a, b):
                        # new Lj
                        nLj = Lj - {v}
                        if not nLj:
                            continue
                        res = list(bicliques)
                        res.pop(j)
                        res.pop(i)
                        res.append((frozenset(a), frozenset(b)))
                        res.append((frozenset(nLj), frozenset(Rj)))
                        return res
                    # v -> Ri
                    a = Li; b = Ri | {v}
                    if a and b and not (a & b) and graph_valid_biclique(a, b):
                        nLj = Lj - {v}
                        if not nLj:
                            continue
                        res = list(bicliques)
                        res.pop(j)
                        res.pop(i)
                        res.append((frozenset(a), frozenset(b)))
                        res.append((frozenset(nLj), frozenset(Rj)))
                        return res
                for v in list(Rj):
                    a = Li; b = Ri | {v}
                    if a and b and not (a & b) and graph_valid_biclique(a, b):
                        nRj = Rj - {v}
                        if not nRj:
                            continue
                        res = list(bicliques)
                        res.pop(j)
                        res.pop(i)
                        res.append((frozenset(a), frozenset(b)))
                        res.append((frozenset(Lj), frozenset(nRj)))
                        return res
                    a = Li | {v}; b = Ri
                    if a and b and not (a & b) and graph_valid_biclique(a, b):
                        nRj = Rj - {v}
                        if not nRj:
                            continue
                        res = list(bicliques)
                        res.pop(j)
                        res.pop(i)
                        res.append((frozenset(a), frozenset(b)))
                        res.append((frozenset(Lj), frozenset(nRj)))
                        return res
        return None

    def polish(bicliques, edge_set, deadline):
        improved = True
        cur = list(bicliques)
        while improved and time.time() < deadline:
            improved = False
            m = try_merge(cur, edge_set, deadline)
            if m is not None:
                cur = m
                improved = True
                continue
            a = try_absorb(cur, edge_set, deadline)
            if a is not None:
                cur = a
                improved = True
        return cur

    def validate_and_cover(bicliques):
        covered = set()
        for (L, R) in bicliques:
            if not L or not R:
                return None
            if L & R:
                return None
            for l in L:
                for r in R:
                    if l == r:
                        return None
                    e = me(l, r)
                    if e not in edge_set:
                        return None
                    if e in covered:
                        return None
                    covered.add(e)
        if covered != edge_set:
            return None
        return bicliques

    best = None
    best_count = total_edges + 1

    strategy_sets = [
        ['expand'],
        ['common'],
        ['degree'],
        ['expand', 'common'],
        ['expand', 'degree'],
        ['common', 'degree'],
        ['expand', 'common', 'degree'],
    ]

    for ss in strategy_sets:
        if time.time() > DEADLINE:
            break
        bic = greedy_partition(B, ss, DEADLINE)
        if validate_and_cover(bic) is None:
            continue
        bic = polish(bic, edge_set, DEADLINE)
        if validate_and_cover(bic) is None:
            continue
        if len(bic) < best_count:
            best_count = len(bic)
            best = bic

    if best is None:
        # fallback: star partition (always valid)
        best = []
        for u in range(n):
            nb = [v for v in adj[u]]
            if nb:
                best.append((frozenset([u]), frozenset(nb)))

    result = []
    for (L, R) in best:
        result.append([list(L), list(R)])

    return result
