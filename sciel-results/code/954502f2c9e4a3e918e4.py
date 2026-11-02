import time
import random
from collections import deque

# EVOLVE-BLOCK-START


def _short_cycle_under(n, adj, g):
    """Return a simple cycle of length < g as a list of vertices, or None."""
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        prev = [-1] * n
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            if dist[u] >= g - 1:
                continue
            for v in adj[u]:
                if v == prev[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    prev[v] = u
                    dq.append(v)
                else:
                    # find LCA of u and v via parent pointers
                    up = {}
                    x = u
                    d = dist[u]
                    while x != -1:
                        up[x] = d
                        x = parent[x]
                        d -= 1
                    x = v
                    d = dist[v]
                    while x != -1:
                        if x in up:
                            L = d + up[x] + 1
                            if L < g:
                                path_u = []
                                y = u
                                while y != x:
                                    path_u.append(y)
                                    y = parent[y]
                                path_u.append(x)
                                path_v = []
                                y = v
                                while y != x:
                                    path_v.append(y)
                                    y = parent[y]
                                return path_u + path_v[::-1]
                            break
                        x = parent[x]
                        d -= 1
    return None


def _verify(n, adj, k, g):
    for u in range(n):
        if len(adj[u]) != k:
            return False
        if u in adj[u]:
            return False
        seen = set()
        for v in adj[u]:
            if v in seen:
                return False
            seen.add(v)
    if _short_cycle_under(n, adj, g) is not None:
        return False
    return True


def _build_edges(adj):
    edges = []
    for u in range(len(adj)):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


def _random_kregular(n, k, rng):
    for _ in range(3000):
        stubs = []
        for u in range(n):
            stubs += [u] * k
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b or b in adj[a]:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if ok and all(len(adj[u]) == k for u in range(n)):
            return [list(s) for s in adj]
    return None


def _swap(adj, u, v, a, b):
    if v not in adj[u] or b not in adj[a]:
        return False
    for (n1, n2, n3, n4) in [(u, a, v, b), (u, b, v, a)]:
        if n1 == n2 or n3 == n4:
            continue
        if n2 in adj[n1] or n4 in adj[n3]:
            continue
        adj[u].remove(v); adj[v].remove(u)
        adj[a].remove(b); adj[b].remove(a)
        adj[n1].append(n2); adj[n2].append(n1)
        adj[n3].append(n4); adj[n4].append(n3)
        return True
    return False


def construct(k, g):
    start = time.time()
    deadline = start + 135.0
    rng = random.Random(7)

    if k != 4:
        n = max(2 * k, k + 2)
        adj = _random_kregular(n, k, rng)
        if adj is None:
            adj = [set() for _ in range(n)]
            for u in range(n):
                for d in range(1, k + 1):
                    v = (u + d) % n
                    if v != u:
                        adj[u].add(v); adj[v].add(u)
            adj = [list(s) for s in adj]
        return n, _build_edges(adj)

    best_n = None
    best_adj = None
    for n in range(250, 331):
        if time.time() > deadline - 8:
            break
        for trial in range(25):
            if time.time() > deadline - 4:
                break
            adj = _random_kregular(n, 4, rng)
            if adj is None:
                continue
            cyc = _short_cycle_under(n, adj, g)
            iters = 0
            while cyc is not None and iters < 3000:
                if time.time() > deadline - 2:
                    break
                iters += 1
                L = len(cyc)
                ei = rng.randrange(L)
                u = cyc[ei]
                v = cyc[(ei + 1) % L]
                a = rng.randrange(n)
                if not adj[a]:
                    continue
                b = rng.choice(adj[a])
                if (a == u and b == v) or (a == v and b == u):
                    continue
                if _swap(adj, u, v, a, b):
                    cyc = _short_cycle_under(n, adj, g)
            if cyc is None and _verify(n, adj, k, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_adj = [list(x) for x in adj]
                return n, _build_edges(adj)

    if best_adj is not None:
        return best_n, _build_edges(best_adj)

    n = 250
    adj = [set() for _ in range(n)]
    for u in range(n):
        for d in (1, 2, 3, 4):
            v = (u + d) % n
            if v != u:
                adj[u].add(v); adj[v].add(u)
    adj = [list(s) for s in adj]
    return n, _build_edges(adj)

# EVOLVE-BLOCK-END
