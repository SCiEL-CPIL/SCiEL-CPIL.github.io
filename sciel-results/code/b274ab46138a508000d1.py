import time
import random
from collections import deque

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges) of a k-regular graph with girth at least 2d+4 (d>=3)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _girth_violation(n, adj, min_girth):
    """Return a short cycle (list of vertices) if one has length < min_girth, else None.
    Uses BFS from each vertex, only exploring to depth min_girth//2 - 1."""
    max_depth = min_girth // 2 - 1
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            du = dist[u]
            if du >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    dq.append(v)
                elif parent[v] != u and parent[u] != v:
                    clen = dist[u] + dist[v] + 1
                    if clen < min_girth:
                        # reconstruct cycle
                        cyc = [u]
                        a = u
                        while a != s:
                            a = parent[a]
                            cyc.append(a)
                        cyc2 = [v]
                        b = v
                        while b != s:
                            b = parent[b]
                            cyc2.append(b)
                        return cyc + cyc2[::-1]
    return None


def _try_switch(adj, edge_set, u, v, x, y, min_girth):
    """Remove edges (u,v) and (x,y); add (u,x) and (v,y) (or the cross pairing) if it
    keeps the graph simple and does not create a cycle shorter than min_girth.
    Returns True if a valid switch was applied (in place)."""
    if u == x or u == y or v == x or v == y:
        return False
    e1 = (u, v) if u < v else (v, u)
    e2 = (x, y) if x < y else (y, x)
    if e1 == e2:
        return False
    # candidate new edges: (u,x)&(v,y) and (u,y)&(v,x)
    for nu, nv in ((u, x), (u, y)):
        for pu, pv in ((v, y), (v, x)):
            if nu == nv or pu == pv:
                continue
            if (nu, nv) == (pu, pv):
                continue
            a1 = (nu, nv) if nu < nv else (nv, nu)
            a2 = (pu, pv) if pu < pv else (pv, pu)
            if a1 in edge_set or a2 in edge_set:
                continue
            if a1 == e1 or a1 == e2 or a2 == e1 or a2 == e2:
                continue
            # apply
            edge_set.discard(e1)
            edge_set.discard(e2)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            adj[nu].append(nv)
            adj[nv].append(nu)
            adj[pu].append(pv)
            adj[pv].append(pu)
            edge_set.add(a1)
            edge_set.add(a2)
            cyc = _girth_violation(len(adj), adj, min_girth)
            if cyc is None:
                return True
            # revert
            edge_set.discard(a1)
            edge_set.discard(a2)
            adj[nu].remove(nv)
            adj[nv].remove(nu)
            adj[pu].remove(pv)
            adj[pv].remove(pu)
            adj[u].append(v)
            adj[v].append(u)
            adj[x].append(y)
            adj[y].append(x)
            edge_set.add(e1)
            edge_set.add(e2)
    return False


def _local_search(n, edges, k, min_girth, deadline):
    """Try to repair short cycles via 2-switches while preserving k-regularity."""
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v) if u < v else (v, u))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        cyc = _girth_violation(n, adj, min_girth)
        if cyc is None:
            return list(edge_set)
        L = len(cyc)
        # for each pair of non-adjacent edges in the cycle, attempt a switch
        tried = set()
        for i in range(L):
            a1 = cyc[i]
            b1 = cyc[(i + 1) % L]
            for j in range(L):
                if j == i or j == (i + 1) % L or j == (i - 1) % L:
                    continue
                a2 = cyc[j]
                b2 = cyc[(j + 1) % L]
                key = frozenset([(a1, b1), (a2, b2)])
                if key in tried:
                    continue
                tried.add(key)
                if _try_switch(adj, edge_set, a1, b1, a2, b2, min_girth):
                    improved = True
                    break
            if improved:
                break
    return list(edge_set)


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130  # safety margin vs 150s kill

    best_n, best_edges = _luw_construct(k, g)

    # Refinement: try to reduce girth violations via 2-switch local search.
    if time.time() < deadline - 20:
        refined = _local_search(best_n, best_edges, k, g, deadline - 20)
        # verify refined graph is still k-regular and has girth >= g
        deg = [0] * best_n
        for u, v in refined:
            deg[u] += 1
            deg[v] += 1
        if all(d == k for d in deg) and len(refined) == best_n * k // 2:
            adj = [[] for _ in range(best_n)]
            for u, v in refined:
                adj[u].append(v)
                adj[v].append(u)
            if _girth_violation(best_n, adj, g) is None:
                best_edges = refined

    return best_n, best_edges
