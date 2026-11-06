"""Greedy maximal-clique partition with deadline guard."""
# EVOLVE-BLOCK-START
import time
import random

def _deadline():
    return time.time() + 140.0

def _random_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
    return adj, edges

def _greedy_partition(adj, edges, deadline, max_clique_size=8):
    cliques = []
    uncovered = set(edges)
    while uncovered and time.time() < deadline:
        # Pick an uncovered edge
        u, v = next(iter(uncovered))
        # Build a clique containing u and v, growing greedily
        clique = {u, v}
        candidates = sorted(adj[u] & adj[v], key=lambda x: -len(adj[x]))
        for cand in candidates:
            if time.time() > deadline:
                break
            if len(clique) >= max_clique_size:
                break
            if cand in clique:
                continue
            if all(cand in adj[c] for c in clique):
                clique.add(cand)
        cliques.append(sorted(clique))
        cl = list(clique)
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                a, b = cl[i], cl[j]
                uncovered.discard((a, b))
                uncovered.discard((b, a))
    for (u, v) in uncovered:
        cliques.append([u, v])
    return cliques

def construct():
    deadline = _deadline()
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        adj, edges = _random_graph(n, p, seed)
        cliques = _greedy_partition(adj, edges, deadline, max_clique_size=8)
        results.append(cliques)
    return results
# EVOLVE-BLOCK-END
