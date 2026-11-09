"""Clique partition via Bron-Kerbosch maximal cliques + greedy set cover."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

_START = time.time()
_LIMIT = 118.0  # safety margin under 120s


def _deadline_ok():
    return (time.time() - _START) < _LIMIT


def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _bron_kerbosch(adj, max_size=8):
    """Find all maximal cliques using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []

    def bk(R, P, X):
        if not _deadline_ok():
            return
        if not P and not X:
            if len(R) >= 2:
                cliques.append(tuple(sorted(R)))
            return
        if len(R) >= max_size:
            if len(R) >= 2:
                cliques.append(tuple(sorted(R)))
            return
        # Choose pivot u from P ∪ X with max |N(u) ∩ P|
        union_px = P | X
        if union_px:
            pivot = max(union_px, key=lambda u: len(adj[u] & P))
            candidates = P - adj[pivot]
        else:
            candidates = P
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bk(new_R, new_P, new_X)
            P.discard(v)
            X.add(v)

    bk(set(), set(range(n)), set())
    return cliques


def _clique_edges(cl):
    """Return set of edges (as frozensets) in clique cl."""
    n = len(cl)
    es = set()
    for i in range(n):
        for j in range(i + 1, n):
            es.add(frozenset((cl[i], cl[j])))
    return es


def _solve_one(n, p, seed):
    edges = _random_edges(n, p, seed)
    edge_set = set(frozenset(e) for e in edges)
    adj = _adjacency(n, edges)

    # Find all maximal cliques of size >= 2
    cliques = _bron_kerbosch(adj, max_size=10)

    # For each clique, compute the set of edges it covers
    clique_edge_sets = []
    for cl in cliques:
        if len(cl) < 2:
            continue
        es = _clique_edges(cl)
        # Only keep edges that actually exist in the graph
        es = es & edge_set
        if es:
            clique_edge_sets.append((cl, es))

    # Greedy set cover: minimize number of cliques
    remaining = set(edge_set)
    chosen = []

    # Sort cliques by size descending as initial heuristic
    # But we do dynamic greedy: at each step pick clique covering most remaining edges
    # For efficiency, maintain active list
    active = [(cl, es) for cl, es in clique_edge_sets if es & remaining]

    while remaining and _deadline_ok():
        best = None
        best_count = 0
        for cl, es in active:
            cov = len(es & remaining)
            if cov > best_count:
                best_count = cov
                best = (cl, es)
                if cov == len(es):
                    # Can't do better than covering all edges of this clique
                    # But a larger clique might cover more... keep going a bit
                    pass
        if best is None:
            # No clique covers any remaining edge; use single-edge cliques
            for e in remaining:
                chosen.append(sorted(e))
            remaining.clear()
            break

        cl, es = best
        chosen.append(list(cl))
        remaining -= es
        # Update active: remove cliques that no longer have all edges in remaining
        # Actually, we can still use a clique if it covers at least one remaining edge
        # But for correctness of "exactly one", we need each edge in exactly one clique.
        # The greedy approach: once we pick a clique, we remove all its edges.
        # A clique can only be picked if ALL its edges are still remaining? No -
        # we just need the clique to be a valid clique in the original graph, and
        # we assign it to cover its edges. But if some edges are already covered,
        # we can't use that clique again. So we should only use cliques whose
        # edges are ALL still remaining.
        active = [(c, e) for c, e in active if e <= remaining]

    # Fallback: leftover edges as single-edge cliques
    for e in remaining:
        chosen.append(sorted(e))

    return chosen


def construct():
    res1 = _solve_one(32, 0.6, 601)
    res2 = _solve_one(64, 0.55, 602)
    return [res1, res2]
# EVOLVE-BLOCK-END
