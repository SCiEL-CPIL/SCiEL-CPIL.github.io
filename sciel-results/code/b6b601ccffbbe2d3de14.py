import time
import random as _rnd
import itertools

_DEADLINE = 150.0
_MARGIN = 12.0


def _gen_instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _maximal_cliques(n, adj, cap=20000):
    # Bron-Kerbosch with pivot, returns list of frozensets (cliques)
    cliques = []
    adj_list = [sorted(s) for s in adj]

    def bk(R, P, X):
        if len(cliques) >= cap:
            return
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # choose pivot
        union = set()
        for v in P:
            union |= adj[v]
        for v in X:
            union |= adj[v]
        if not union:
            candidates = list(P)
        else:
            candidates = [v for v in P if not (adj[v] & union)]
        for v in candidates:
            bk(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}
            if len(cliques) >= cap:
                return

    bk(frozenset(), set(range(n)), frozenset())
    return cliques


def _clique_edges(clique, adj):
    # yield edges inside clique
    lst = sorted(clique)
    for i in range(len(lst)):
        a = lst[i]
        for b in lst[i + 1:]:
            if b in adj[a]:
                yield (a, b)


def _greedy_cover(n, adj, edge_set, cliques, rng, max_iters=200000):
    """Randomized greedy set cover. Returns list of cliques (frozenset) or None if fail."""
    uncovered = set(edge_set)
    chosen = []
    # precompute edge count per clique (only edges that exist)
    # We'll maintain per-clique remaining uncovered edges lazily.
    # For efficiency, represent cliques as frozensets; compute intersection size on the fly but cache.
    # Use a dict clique -> set of its edges
    cliq_edges = {}
    for c in cliques:
        e = set(_clique_edges(c, adj))
        if e:
            cliq_edges[c] = e
    # work list
    import heapq
    # We'll do greedy: each step pick clique maximizing |clique_edges & uncovered|, with random tie-break.
    # To keep it fast, limit candidate cliques to those with size>=2.
    cand = list(cliq_edges.keys())
    while uncovered:
        best = None
        best_cnt = 0
        best_key = None
        # sample-based: if cand large, sample a subset each step to save time
        if len(cand) > 400:
            sample = cand if len(cand) <= 400 else rng.sample(cand, 400)
        else:
            sample = cand
        for c in sample:
            ce = cliq_edges[c]
            cnt = len(ce & uncovered)
            if cnt > best_cnt:
                best_cnt = cnt
                best = c
                best_key = (cnt, rng.random())
            elif cnt == best_cnt and cnt > 0:
                k = (cnt, rng.random())
                if k > best_key:
                    best_key = k
                    best = c
        if best is None or best_cnt == 0:
            return None
        uncovered -= cliq_edges[best]
        chosen.append(best)
        # remove best from cand
        try:
            cand.remove(best)
        except ValueError:
            pass
    return chosen


def _local_improve(n, adj, edge_set, chosen, cliq_edges, rng, max_rounds=200):
    """Try to reduce clique count by merging: find a clique covering union of two chosen cliques' edges
    that is disjoint from rest. Also try replacing a chosen clique with a larger one if it still covers
    same edges subset (no gain). Main gain: find clique that is superset of a chosen clique and disjoint
    from others -> no count change. So focus on 2-merge: union of edges of two cliques equals edges of one
    bigger clique, and that bigger clique's edges are subset of (their union) i.e. exactly equal, and disjoint
    from other chosen cliques' edges."""
    chosen = list(chosen)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        # build edge->clique membership for chosen
        chosen_edges = [set(cliq_edges[c]) for c in chosen]
        # try all pairs
        m = len(chosen)
        for i in range(m):
            for j in range(i + 1, m):
                union_edges = chosen_edges[i] | chosen_edges[j]
                # find a clique c such that set(cliq_edges[c]) == union_edges and c not in chosen (or is)
                # and c's edges disjoint from other chosen edges (they are, since union subset of i|j)
                # We need c covering exactly union_edges.
                for c in cliq_edges:
                    if c in chosen:
                        continue
                    if cliq_edges[c] == union_edges:
                        # replace i,j with c
                        new_chosen = [c] + [chosen[k] for k in range(m) if k not in (i, j)]
                        # verify disjointness and full coverage
                        ok = True
                        seen = set()
                        for cc in new_chosen:
                            e = cliq_edges[cc]
                            if e & seen:
                                ok = False
                                break
                            seen |= e
                        if ok and seen == edge_set:
                            chosen = new_chosen
                            chosen_edges = [set(cliq_edges[c]) for c in chosen]
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return chosen


def _solve_instance(n, p, seed, time_budget):
    edges = _gen_instance(n, p, seed)
    edge_set = set(edges)
    adj = _build_adj(n, edges)
    cliques = _maximal_cliques(n, adj)
    if not cliques:
        # fallback: single edges
        return [[list(e) for e in edges]]
    cliq_edges = {}
    for c in cliques:
        e = set(_clique_edges(c, adj))
        if e:
            cliq_edges[c] = e
    # sort cliques by size desc for tie-breaking preference
    cliques_sorted = sorted(cliques, key=lambda c: (-len(c), c))
    best = None
    best_len = len(edge_set)  # upper bound: single edges
    rng = _rnd.Random(12345)
    t0 = time.time()
    restart = 0
    while time.time() - t0 < time_budget:
        restart += 1
        sol = _greedy_cover(n, adj, edge_set, cliques_sorted, rng)
        if sol is None:
            continue
        # local improve
        sol = _local_improve(n, adj, edge_set, sol, cliq_edges, rng, max_rounds=30)
        if len(sol) < best_len:
            best_len = len(sol)
            best = sol
            # if optimal-ish (e.g., very small), stop
            if best_len <= 1:
                break
        # early stop if we have a great answer and time is tight
        if best_len <= 2 and time.time() - t0 > time_budget * 0.5:
            break
    if best is None:
        return [[list(e) for e in edges]]
    out = []
    for c in best:
        out.append(sorted(c))
    return out


def construct():
    t_start = time.time()
    total_budget = _DEADLINE - _MARGIN
    # allocate budget per instance
    b1 = total_budget * 0.45
    b2 = total_budget - b1
    r1 = _solve_instance(32, 0.6, 601, b1)
    elapsed = time.time() - t_start
    b2 = max(1.0, total_budget - elapsed)
    r2 = _solve_instance(64, 0.55, 602, b2)
    return [r1, r2]
