"""Small vertex sets meeting every maximal clique of three fixed random graphs."""
# EVOLVE-BLOCK-START
import time

_DEADLINE = 150.0


def _elapsed():
    return time.time() - _T0


def _build_graph(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _maximal_cliques(adj):
    n = len(adj)
    adjb = [set(a) for a in adj]
    cliques = []
    P = set(range(n))
    X = set()
    R = []

    def bronk(P, X, R):
        while P or X:
            u = max(P | X, key=lambda x: len(P & adjb[x]))
            Pn = P & adjb[u]
            bronk(Pn, X & adjb[u], R + [u])
            P.discard(u)
            X.add(u)
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))

    bronk(P, X, R)
    return cliques


def _solve(adj, cliques):
    n = len(adj)
    allv = set(range(n))
    # map vertex -> set of cliques it hits
    vcl = [set() for _ in range(n)]
    for i, cl in enumerate(cliques):
        for v in cl:
            vcl[v].add(i)

    # cliques that need covering (all have >=2 by construction)
    m = len(cliques)
    if m == 0:
        return []

    # initial lower bound: sum of clique sizes / coverage heuristic
    # better: greedy set-cover lower bound via LP relaxation is complex; use greedy
    def greedy_lower(remaining):
        # lower bound = max over cliques of 1 if non-empty (each needs >=1)
        # stronger: use a simple packing bound
        return len(remaining) if remaining else 0

    # We'll do B&B with propagation
    best = [None]
    best_size = [n + 1]

    # Precompute for each clique the list of vertices in it
    cliq_list = [sorted(c) for c in cliques]

    # Order cliques by size (smallest first) for propagation
    # We track which cliques are still "open" (not yet hit)
    # A clique is hit if any selected vertex is in it.

    # State: selected set, open cliques (frozenset of indices)
    # We use recursion with a chosen clique.

    # For speed, represent open cliques as a frozenset of ints.

    def cover_check(sel, open_cl):
        # nothing to do
        return

    # We'll implement iterative deepening-ish B&B:
    # pick most constrained open clique (fewest candidate vertices not yet excluded)
    # branch on each vertex in it.

    # To bound, use: |selected| + lower_bound >= best_size -> prune.
    # lower_bound: each open clique needs at least 1 new vertex, but one vertex can
    # cover multiple. Use a simple bound: number of open cliques that are pairwise
    # disjoint (packing). Compute a greedy packing.

    def packing_bound(open_cl):
        # greedy: repeatedly pick the smallest open clique, remove all cliques it intersects
        # each picked clique needs a distinct vertex (since disjoint)
        if not open_cl:
            return 0
        import random
        # deterministic: sort by size
        remaining = sorted(open_cl, key=lambda i: len(cliq_list[i]))
        count = 0
        used = set()
        for i in remaining:
            if i in used:
                continue
            count += 1
            # remove all cliques that share a vertex with cliq i
            ci = cliq_list[i]
            for j in remaining:
                if j in used or j == i:
                    continue
                if cliq_list[j] & set(ci):
                    used.add(j)
        return count

    def dfs(sel, open_cl, depth):
        if _elapsed() > _DEADLINE - 5:
            return
        if not open_cl:
            if len(sel) < best_size[0]:
                best_size[0] = len(sel)
                best[0] = list(sel)
            return
        # bound
        lb = packing_bound(open_cl)
        if len(sel) + lb >= best_size[0]:
            return
        # pick most constrained open clique
        # candidate vertices = union of vertices in open clique that are not yet selected
        # most constrained = smallest number of available vertices
        best_cl = None
        best_cand = None
        best_csize = 10**9
        for i in open_cl:
            cand = [v for v in cliq_list[i] if v not in sel]
            if not cand:
                # infeasible
                return
            if len(cand) < best_csize:
                best_csize = len(cand)
                best_cl = i
                best_cand = cand
                if best_csize == 1:
                    break
        # branch
        # order candidates by number of cliques they cover (descending)
        cand_sorted = sorted(best_cand, key=lambda v: -len(vcl[v] & open_cl))
        for v in cand_sorted:
            new_sel = sel | {v}
            # update open cliques
            new_open = set()
            for i in open_cl:
                if v in cliq_list[i]:
                    continue
                new_open.add(i)
            dfs(new_sel, new_open, depth + 1)

    # Initial greedy solution to set best
    def greedy():
        sel = set()
        open_cl = set(range(m))
        while open_cl:
            # pick vertex covering most open cliques
            best_v = None
            best_cov = -1
            # consider vertices in the smallest open clique
            sc = min(open_cl, key=lambda i: len(cliq_list[i]))
            for v in cliq_list[sc]:
                if v in sel:
                    continue
                cov = len(vcl[v] & open_cl)
                if cov > best_cov:
                    best_cov = cov
                    best_v = v
            if best_v is None:
                break
            sel.add(best_v)
            new_open = set()
            for i in open_cl:
                if best_v in cliq_list[i]:
                    continue
                new_open.add(i)
            open_cl = new_open
        return sel

    g = greedy()
    best_size[0] = len(g)
    best[0] = list(g)

    # Now B&B to improve
    dfs(set(), set(range(m)), 0)

    if best[0] is None:
        return []
    return sorted(best[0])


def construct():
    global _T0
    _T0 = time.time()
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if _elapsed() > _DEADLINE - 10:
            # not enough time; return a valid-but-possibly-suboptimal fallback
            # fallback: all vertices (valid, huge)
            out.append(list(range(n)))
            continue
        adj = _build_graph(n, p, seed)
        cliques = _maximal_cliques(adj)
        sol = _solve(adj, cliques)
        out.append(sol)
    return out
# EVOLVE-BLOCK-END
