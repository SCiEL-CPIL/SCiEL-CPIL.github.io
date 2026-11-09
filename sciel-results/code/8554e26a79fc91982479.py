"""Clique partition of two fixed dense random graphs."""
import time
from random import Random

def _random_graph(n, p, seed):
    rng = Random(seed)
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return edges

def _maximal_cliques(adj, R, P, X, out, node_limit, nodes):
    if not P and not X:
        if len(R) >= 2:
            out.append(frozenset(R))
        nodes[0] += 1
        return nodes[0] <= node_limit
    Pset = set(P)
    Xset = set(X)
    cand = Pset | Xset
    if not cand:
        nodes[0] += 1
        return nodes[0] <= node_limit
    pivot = max(cand, key=lambda v: len(adj[v] & Pset))
    candidates = [v for v in Pset if v not in adj[pivot]]
    for v in candidates:
        nodes[0] += 1
        if nodes[0] > node_limit:
            return False
        newR = R | {v}
        newP = Pset & adj[v]
        newX = Xset & adj[v]
        ok = _maximal_cliques(adj, newR, newP, newX, out, node_limit, nodes)
        if not ok:
            return False
        Pset.discard(v)
        Xset.add(v)
    return True

def _best_clique_for_edges(edges, node_limit=6000):
    if not edges:
        return None
    verts = set()
    for u, v in edges:
        verts.add(u); verts.add(v)
    sub_adj = {v: set() for v in verts}
    for u, v in edges:
        sub_adj[u].add(v)
        sub_adj[v].add(u)
    out = []
    nodes = [0]
    _maximal_cliques(sub_adj, frozenset(), set(verts), set(), out, node_limit, nodes)
    best = None
    best_cnt = 0
    for cl in out:
        if len(cl) < 2:
            continue
        clset = set(cl)
        cnt = 0
        for u, v in edges:
            if u in clset and v in clset:
                cnt += 1
        if cnt > best_cnt:
            best_cnt = cnt
            best = cl
    if best is None or best_cnt < 1:
        u, v = next(iter(edges))
        return {u, v}
    return best

def _partition_cliques(edges, deadline):
    remaining = set(edges)
    cliques = []
    while remaining:
        if time.time() > deadline - 2.0:
            break
        cl = _best_clique_for_edges(remaining)
        if cl is None or len(cl) < 2:
            for e in remaining:
                cliques.append(set(e))
            remaining = set()
            break
        clset = cl
        remaining = {e for e in remaining if not (e[0] in clset and e[1] in clset)}
        cliques.append(clset)
    for e in remaining:
        cliques.append(set(e))
    return cliques

def construct():
    deadline = time.time() + 140.0
    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = _random_graph(n, p, seed)
        cliques = _partition_cliques(edges, deadline)
        results.append([list(c) for c in cliques])
    return results
