"""Search for a dense triangle-free, non-3-colorable graph on 16 vertices."""
import time
import random

_N = 16
_ALL = (1 << _N) - 1

def _is_triangle_free(adj):
    for u in range(_N):
        a = adj[u]
        for v in range(u + 1, _N):
            if a & (1 << v):
                if adj[v] & a:
                    return False
    return True

def _is_3colorable(adj):
    colors = [-1] * _N
    sat = [0] * _N
    forbid = [0] * _N
    deg = [0] * _N
    for u in range(_N):
        deg[u] = adj[u].bit_count()

    def assign(u, c):
        colors[u] = c
        for w in range(_N):
            if adj[u] & (1 << w):
                if colors[w] >= 0:
                    sat[w] += 1
                else:
                    forbid[w] |= (1 << c)
                    if forbid[w] == 7:
                        return False
        return True

    def unassign(u, c):
        colors[u] = -1
        for w in range(_N):
            if adj[u] & (1 << w):
                if colors[w] >= 0:
                    sat[w] -= 1
                else:
                    forbid[w] &= ~(1 << c)
        return True

    def dsatur():
        best = -1
        best_sat = -1
        best_deg = -1
        for u in range(_N):
            if colors[u] < 0:
                s = sat[u]
                if s > best_sat or (s == best_sat and deg[u] > best_deg):
                    best_sat = s
                    best_deg = deg[u]
                    best = u
        if best < 0:
            return True
        for c in (1, 2, 4):
            if not (forbid[best] & c):
                if assign(best, c):
                    if dsatur():
                        return True
                unassign(best, c)
        return False

    return dsatur()

def _edge_count(adj):
    return sum(adj[u].bit_count() for u in range(_N)) // 2

def _mylevel(n):
    n2 = 2 * n + 1
    edges = []
    for i in range(n):
        edges.append([i, (i + 1) % n])
    for i in range(n):
        edges.append([i, n + i])
        edges.append([(i + 1) % n, n + i])
    for i in range(n):
        edges.append([n + i, n2 - 1])
    return edges

def construct():
    deadline = time.time() + 125.0
    rng = random.Random(777)

    best_adj = None
    best_count = 0

    # Fallback: Grötzsch graph (11 verts) + 5 isolated
    fb = _mylevel(5)
    fb_adj = [0] * _N
    for u, v in fb:
        fb_adj[u] |= 1 << v
        fb_adj[v] |= 1 << u
    best_adj = fb_adj
    best_count = len(fb)

    # Simulated annealing / local search to maximize edges, triangle-free.
    # Start from random triangle-free graphs, add/remove/flip edges.
    for restart in range(500):
        if time.time() >= deadline:
            break
        # Initialize: random graph, then remove triangles
        adj = [0] * _N
        for u in range(_N):
            for v in range(u + 1, _N):
                if rng.random() < 0.3:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        # Remove all triangles
        changed = True
        while changed:
            changed = False
            for u in range(_N):
                a = adj[u]
                for v in range(u + 1, _N):
                    if a & (1 << v):
                        common = a & adj[v]
                        if common:
                            # remove one edge to break triangle
                            w = (common & -common).bit_length() - 1
                            adj[u] &= ~(1 << w)
                            adj[w] &= ~(1 << u)
                            changed = True
        if not _is_triangle_free(adj):
            continue
        cur_count = _edge_count(adj)
        if cur_count > best_count and not _is_3colorable(adj):
            best_adj = adj[:]
            best_count = cur_count

        # Local search: try adding edges, flipping, etc.
        for step in range(200):
            if time.time() >= deadline:
                break
            # Try to add a random missing edge
            u = rng.randrange(_N)
            v = rng.randrange(_N)
            if u == v:
                continue
            if adj[u] & (1 << v):
                continue
            if adj[u] & adj[v]:
                continue  # would create triangle
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            new_count = cur_count + 1
            if new_count > best_count and not _is_3colorable(adj):
                best_adj = adj[:]
                best_count = new_count
            cur_count = new_count

    def _edges_from_adj(a):
        E = []
        for u in range(_N):
            for v in range(u + 1, _N):
                if a[u] & (1 << v):
                    E.append([u, v])
        return E

    return _edges_from_adj(best_adj)
