"""Clique partitions of split graphs: structural set-cover + local refinement."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    _T0 = time.time()
    _DEADLINE = 135.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def _max_clique(adj, cand, cap=None):
        """Bron-Kerbosch with pivot; returns one maximum clique (list of ints)."""
        cand = list(cand)
        if not cand:
            return []
        best = []

        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            U = P | X
            if U:
                piv = max(U, key=lambda v: len(adj[v] & P))
                cand2 = [v for v in P if v not in adj[piv]]
            else:
                cand2 = list(P)
            for v in cand2:
                R.append(v)
                bk(R, P & adj[v], X & adj[v])
                R.pop()
                P.discard(v)
                X.add(v)

        bk([], set(cand), set())
        return best

    def _solve_one(n, k, p, seed, budget):
        # ---- build adjacency (split structure) ----
        C = list(range(k))
        I = list(range(k, n))
        N = {v: set() for v in I}
        rng = random.Random(seed)
        for v in I:
            for u in C:
                if rng.random() < p:
                    N[v].add(u)

        # ---- greedy set cover: C-I edges (family N(v)) then C-C edges (family of C-subsets) ----
        used = [[False] * n for _ in range(n)]
        cliques = []
        for v in I:
            if N[v]:
                cliques.append(sorted(list(N[v]) + [v]))
                for u in N[v]:
                    for w in N[v]:
                        if u < w:
                            used[u][w] = True

        rem = set()
        for i in range(k):
            for j in range(i + 1, k):
                if not used[i][j]:
                    rem.add((i, j))

        while rem:
            best = None
            best_cov = -1
            for (i, j) in rem:
                cov = 0
                for (x, y) in rem:
                    if x in (i, j) and y in (i, j):
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best = (i, j)
            i, j = best
            cliques.append([i, j])
            rem.discard((i, j))

        # ---- local search: merge / split / swap ----
        def _edge_count(cl):
            c = 0
            L = len(cl)
            for a in range(L):
                for b in range(a + 1, L):
                    c += 1
            return c

        def _try_merge(i, j):
            A = cliques[i]
            B = cliques[j]
            merged = sorted(set(A) | set(B))
            if len(merged) < 2:
                return False
            if _edge_count(merged) == _edge_count(A) + _edge_count(B):
                cliques[i] = merged
                del cliques[j]
                return True
            return False

        def _try_split(i):
            A = cliques[i]
            if len(A) < 4:
                return False
            for s in range(2, len(A) - 1):
                if s == len(A) - 1:
                    continue
                part1 = A[:s]
                part2 = A[s:]
                if len(part1) < 2 or len(part2) < 2:
                    continue
                if _edge_count(part1) + _edge_count(part2) == _edge_count(A):
                    cliques[i] = part1
                    cliques.insert(i + 1, part2)
                    return True
            return False

        def _try_swap(i, j):
            A = cliques[i]
            B = cliques[j]
            if len(A) < 2 or len(B) < 2:
                return False
            for a in A:
                for b in B:
                    if a == b:
                        continue
                    A2 = (set(A) - {a}) | {b}
                    B2 = (set(B) - {b}) | {a}
                    if len(A2) < 2 or len(B2) < 2:
                        continue
                    if _edge_count(sorted(A2)) + _edge_count(sorted(B2)) == _edge_count(A) + _edge_count(B):
                        cliques[i] = sorted(A2)
                        cliques[j] = sorted(B2)
                        return True
            return False

        improved = True
        while improved and (time.time() - _T0) < budget:
            improved = False
            m = len(cliques)
            for i in range(m):
                for j in range(i + 1, m):
                    if _try_merge(i, j):
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                for i in range(len(cliques)):
                    if _try_split(i):
                        improved = True
                        break
            if not improved:
                m = len(cliques)
                for i in range(m):
                    for j in range(i + 1, m):
                        if _try_swap(i, j):
                            improved = True
                            break
                    if improved:
                        break

        # ---- validate and repair ----
        all_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_edges.add((u, w))
        for v in I:
            for u in N[v]:
                all_edges.add((u, v))
        eid = {e: i for i, e in enumerate(all_edges)}
        count = [0] * len(all_edges)
        valid = True
        for cl in cliques:
            L = len(cl)
            for a in range(L):
                ua = cl[a]
                for b in range(a + 1, L):
                    vb = cl[b]
                    idx = eid.get((min(ua, vb), max(ua, vb)))
                    if idx is None:
                        valid = False
                        break
                    count[idx] += 1
                if not valid:
                    break
            if not valid:
                break
        if valid:
            for c in count:
                if c != 1:
                    valid = False
                    break

        if not valid:
            cliques = [sorted(list(e)) for e in sorted(all_edges)]

        return cliques

    # ---- time budget across the three instances ----
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    total_budget = _DEADLINE - (time.time() - _T0) - 3.0
    per = total_budget / 3.0

    result = []
    for spec in specs:
        if (time.time() - _T0) > _DEADLINE:
            break
        n, k, p, seed = spec
        cliques = _solve_one(n, k, p, seed, per)
        result.append(cliques)

    # ---- safety fallback (one clique per edge) for any missing instance ----
    while len(result) < 3:
        n, k, p, seed = specs[len(result)]
        rng = random.Random(seed)
        cliques = []
        for u in range(k):
            for w in range(u + 1, k):
                cliques.append([u, w])
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cliques.append([u, v])
        result.append(cliques)

    # ---- type normalization: ints must be Python int, sequences must be list ----
    final = []
    for clique_list in result:
        final.append([list(int(v) for v in clique) for clique in clique_list])

    return final[:3]
# EVOLVE-BLOCK-END
