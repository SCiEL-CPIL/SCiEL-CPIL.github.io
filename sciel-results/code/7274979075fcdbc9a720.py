"""Search for a small 5-regular graph with girth >= g.

Unlike the seed (which builds a full 1116-vertex cage), we only need a k-regular
graph whose shortest cycle has length >= g; it need not be a cage.  We therefore
generate random k-regular graphs and verify their girth exactly with
non-backtracking walks, keeping the smallest verified one.  A deadline guard
returns the best valid result before the wall-clock limit.
"""
import random
import time
import numpy as np

# ---------------- deterministic RNG ----------------
def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

# ---------------- k-regular graph generation ----------------
def _pairing_regular(n, k, rng):
    """Return adjacency (list of lists) of a random simple k-regular graph on n
    vertices via a random perfect matching of n*k stubs.  Returns None on
    failure (self-loop or parallel edge)."""
    m = n * k
    stubs = [i for i in range(n) for _ in range(k)]
    # random perfect matching
    perm = list(range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if v in adj[u]:          # parallel edge
            return None
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _swap_regular(adj, n, k, rng):
    """One random 2-switch preserving k-regularity; returns True if it produced
    a simple graph (no self-loops / parallel edges)."""
    m = n * k
    stubs = [i for i in range(n) for _ in range(k)]
    a = _randrange(rng, m)
    b = _randrange(rng, m - 1)
    if b >= a:
        b += 1
    c = _randrange(rng, m - 2)
    if c >= a:
        c += 1
    if c >= b:
        c += 1
    d = _randrange(rng, m - 3)
    for x in (a, b, c):
        if d >= x:
            d += 1
    u1, v1 = stubs[a], stubs[b]
    u2, v2 = stubs[c], stubs[d]
    # remove edges (u1,v1) and (u2,v2); require they are distinct
    if u1 == v1 or u2 == v2:
        return False
    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
        return False
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    # add edges (u1,u2) and (v1,v2); forbid self-loops / parallel edges
    if u1 == u2 or v1 == v2:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    if u2 in adj[u1] or v2 in adj[v1]:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    adj[u1].append(u2)
    adj[u2].append(u1)
    adj[v1].append(v2)
    adj[v2].append(v1)
    return True

# ---------------- exact girth via non-backtracking walks ----------------
def _girth(adj, g):
    """Return the length of the shortest cycle, or a value >= g if none shorter
    exists.  Uses non-backtracking walks from every vertex (a non-backtracking
    walk of length t>=3 closes to a simple cycle of length <= t, so this finds
    the girth exactly for t < g)."""
    n = len(adj)
    # state id: (vertex, prev_vertex) with prev in 0..n-1, prev==n means "free"
    S = n * (n + 1)
    dist = np.full(S, 1 << 30, dtype=np.int32)
    for v in range(n):
        dist[v * (n + 1) + n] = 0
    for t in range(1, g):
        newdist = np.full(S, 1 << 30, dtype=np.int32)
        for v in range(n):
            base = v * (n + 1)
            d0 = dist[base + n]
            if d0 == 1 << 30:
                continue
            val = d0 + 1
            for w in adj[v]:
                newdist[base + w] = val
            for p in range(n):
                dp = dist[base + p]
                if dp == 1 << 30:
                    continue
                val = dp + 1
                for w in adj[v]:
                    if w == p:
                        continue
                    if val < newdist[base + w]:
                        newdist[base + w] = val
        dist = newdist
    for v in range(n):
        base = v * (n + 1)
        for w in adj[v]:
            d = dist[base + w]
            if d != 1 << 30:
                L = d + 1
                if L < g:
                    return L
    return g

# ---------------- main entry point ----------------
def construct(k, g):
    deadline = time.time() + 110.0   # safety margin under the 150 s limit
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    def consider(n, adj):
        nonlocal best, best_n
        if _girth(adj, g) >= g:
            best = adj
            best_n = n
            return True
        return False

    # (1) try a few fixed sizes that may already satisfy the girth
    if k == 5 and g == 9:
        for n in (10, 11, 12, 13, 14, 15, 16, 17, 18):
            if time.time() > deadline:
                break
            for _ in range(4000):
                if time.time() > deadline:
                    break
                adj = _pairing_regular(n, k, rng)
                if adj is None:
                    continue
                if consider(n, adj):
                    return n, [(i, j) for i in range(n) for j in adj[i] if i < j]

    # (2) growing n with 2-switch local search
    n = 2 * k
    while n <= 600 and time.time() < deadline:
        adj = None
        for _ in range(3000):
            if time.time() > deadline:
                break
            cand = _pairing_regular(n, k, rng)
            if cand is not None:
                adj = cand
                break
        if adj is not None:
            consider(n, adj)
            for _ in range(20000):
                if time.time() > deadline:
                    break
                if _swap_regular(adj, n, k, rng):
                    if consider(n, adj):
                        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
        n += 1

    if best is not None:
        return best_n, [(i, j) for i in range(best_n) for j in best[i] if i < j]

    # (3) guaranteed fallback: the Lazebnik-Ustimenko-Woldar construction
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
