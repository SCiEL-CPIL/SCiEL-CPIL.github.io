import time
import random
from collections import defaultdict

def _maximal_cliques(adj, n):
    cliques = []
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # pivot: vertex in P|X with max |P ∩ N(v)|
        PvX = P | X
        best = -1
        pivot = None
        for v in PvX:
            c = 0
            for u in P:
                if v in adj[u]:
                    c += 1
            if c > best:
                best = c
                pivot = v
        if pivot is None:
            for u in list(P):
                P.remove(u)
                bk(R | {u}, P & adj[u], X & adj[u])
                X.add(u)
        else:
            for u in list(P - adj[pivot]):
                P.remove(u)
                bk(R | {u}, P & adj[u], X & adj[u])
                X.add(u)
    bk(set(), set(range(n)), set())
    return cliques

def _min_vertex_cover(adj, deadline):
    n = len(adj)
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    if not edges:
        return []

    # --- Incumbent from the "touched vertices" idea (valid, often small) ---
    touched = set()
    for u, v in edges:
        touched.add(u)
        touched.add(v)
    best_cover = set(touched)
    best_size = len(best_cover)

    # --- Exact branch-and-bound (vertex cover of the edge graph) ---
    def solve(remaining_edges, current_cover):
        nonlocal best_cover, best_size
        if time.time() > deadline:
            return
        if not remaining_edges:
            if len(current_cover) < best_size:
                best_size = len(current_cover)
                best_cover = set(current_cover)
            return

        # Lower bound: greedy matching
        matched = set()
        lb = 0
        for u, v in remaining_edges:
            if u not in matched and v not in matched:
                matched.add(u)
                matched.add(v)
                lb += 1
        if len(current_cover) + lb >= best_size:
            return

        # Branch on the edge with max endpoint-degree-sum
        deg_rem = defaultdict(int)
        for u, v in remaining_edges:
            deg_rem[u] += 1
            deg_rem[v] += 1
        max_score = -1
        branch_edge = None
        for u, v in remaining_edges:
            s = deg_rem[u] + deg_rem[v]
            if s > max_score:
                max_score = s
                branch_edge = (u, v)
        if not branch_edge:
            return
        u, v = branch_edge

        solve({e for e in remaining_edges if u not in e}, current_cover | {u})
        solve({e for e in remaining_edges if v not in e}, current_cover | {v})

    solve(set(edges), set())
    return best_cover

def _local_search(cover, cliques, deadline):
    """2-opt local search minimizing the number of maximal cliques (size>=2) uncovered."""
    cover = set(cover)
    if not cliques:
        return cover

    # occ[v] = list of clique indices containing v
    occ = defaultdict(list)
    for ci, c in enumerate(cliques):
        for v in c:
            occ[v].append(ci)

    # uncovered[i] = True if clique i has no member in cover
    uncovered = [True] * len(cliques)
    for v in cover:
        for ci in occ[v]:
            uncovered[ci] = False

    def count_uncovered():
        c = 0
        for i in range(len(cliques)):
            if uncovered[i]:
                c += 1
        return c

    best_unc = count_uncovered()
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in sorted(cover):
            if time.time() > deadline:
                break
            # tentatively remove v
            cover.remove(v)
            for ci in occ[v]:
                if not uncovered[ci]:
                    uncovered[ci] = True
            unc = count_uncovered()
            if unc <= best_unc:
                best_unc = unc
                improved = True
            else:
                cover.add(v)
                for ci in occ[v]:
                    uncovered[ci] = False
    return cover

def construct():
    start_total = time.time()
    HARD = 145.0          # safety margin before 150 s wall-clock kill
    results = []

    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        # --- Build graph (identical RNG sequence to the seed program) ---
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)

        # --- Fast fallback: touched vertices (always valid) ---
        touched = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    touched.add(u)
                    touched.add(v)
        fallback = sorted(touched)

        remaining_time = HARD - (time.time() - start_total)
        if remaining_time < 8:
            results.append(fallback)
            continue

        # --- Enumerate maximal cliques (Bron-Kerbosch) for the true objective ---
        cliques = _maximal_cliques(adj, n)

        # --- Exact vertex-cover B&B with a time budget ---
        bb_deadline = time.time() + remaining_time * 0.6
        cover = _min_vertex_cover(adj, bb_deadline)

        # --- Local search on the real objective (clique transversal) ---
        ls_deadline = min(time.time() + remaining_time * 0.3, HARD - 3.0)
        cover = _local_search(cover, cliques, ls_deadline)

        # --- Validate; fall back if somehow invalid ---
        ok = True
        for c in cliques:
            if not (c & cover):
                ok = False
                break
        if not ok:
            cover = set(fallback)

        results.append(sorted(cover))

    return results
