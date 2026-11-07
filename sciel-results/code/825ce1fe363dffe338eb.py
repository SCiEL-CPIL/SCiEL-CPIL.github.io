"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 15}

def _deg_count(edges):
    d = {}
    for u, v in edges:
        d[u] = d.get(u, 0) + 1
        d[v] = d.get(v, 0) + 1
    return d

def _is_valid(edges, n):
    # no loops / duplicates
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
    deg = _deg_count(edges)
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # every simple 3-edge path s-u-v-t, s != t, endpoints must have different degree
    for u in range(n):
        for v in adj[u]:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _edge_count(edges):
    return len(edges)

def _complete_bipartite(n, a, b):
    # vertices 0..a-1 are "low" side, a..a+b-1 are "high" side; rest isolated
    edges = []
    for i in range(a):
        for j in range(a, a + b):
            edges.append([i, j])
    return edges

def construct():
    deadline = time.time() + 80.0
    n = P['n']
    best = None
    best_edges = -1

    # Bipartite family: K_{a,b} with a+b = n, a != b  (endpoints always same side,
    # low-side deg = b, high-side deg = a, distinct since a != b -> constraint holds)
    for a in range(1, n):
        b = n - a
        if a == b:
            continue
        edges = _complete_bipartite(n, a, b)
        if _is_valid(edges, n):
            c = _edge_count(edges)
            if c > best_edges:
                best_edges = c
                best = edges
        if time.time() > deadline:
            break

    # Safety fallback: the seed star is always valid (14 edges)
    if best is None or best_edges < 14:
        star = [[0, v] for v in range(1, n)]
        if _is_valid(star, n):
            best = star
            best_edges = len(star)

    # Return as clean Python int lists
    out = []
    for e in best:
        out.append([int(e[0]), int(e[1])])
    return out
# EVOLVE-BLOCK-END
