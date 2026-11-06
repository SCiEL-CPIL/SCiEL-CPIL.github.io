"""C6-free graph on 40 vertices maximizing edges; bitset-validated greedy + local search."""
import time
import random
import numpy as np

N = 40
MASK = (1 << N) - 1
# Precompute per-vertex "neighbors-of-neighbor" masks (distance-2 in current graph).
# d2[i] = union over j in adj[i] of adj[j] (i itself not included).
D2 = [0] * N

def _recompute_d2(adj):
    for i in range(N):
        m = 0
        ai = adj[i]
        # iterate set bits
        x = ai
        while x:
            lb = x & (-x)
            j = lb.bit_length() - 1
            m |= adj[j]
            x ^= lb
        m &= ~ai
        m &= ~(1 << i)
        D2[i] = m

def _would_create_c6(u, v, adj):
    """Return True if adding edge (u,v) creates a C6.
    A C6 through (u,v) is u - a - b - c - d - v with a in N(u), d in N(v),
    and a-b-c-d a length-3 path in G. This exists iff some a in N(u) and d in N(v)
    have a common neighbor b with a common neighbor c (i.e. a ~ b ~ c ~ d,
    b in D2[a] and b in D2[d] with b != a,d). Equivalently: there is a vertex b
    that is at distance 1 from some a in N(u) and distance 2 from some d in N(v)
    via c. Fastest: check if any a in N(u) and d in N(v) share a length-2 path a-c-d
    AND a-b exists... 
    Simpler correct test: C6 = u,a,b,c,d,v. Need a in N(u), d in N(v), and a path
    a-b-c-d of length 3. a-b-c-d exists iff there is c with (c in N(a) and c in D2[d])
    and c != a,d, plus b in N(a) intersect N(c) ... 
    Use: exists a in N(u), d in N(v) such that N(a) & D2[d] != {a} (a length-2 path a-c-d)
    AND there is b in N(a) & N(c). Combine: for each a in N(u), for each d in N(v),
    check if there exists c in (N(a) & D2[d]) \\ {a,d} with (N(a) & N(c) & ~{a,c}) != 0.
    """
    au = adj[u] & ~(1 << v)
    av = adj[v] & ~(1 << u)
    # iterate a in N(u)
    x = au
    while x:
        lb = x & (-x)
        a = lb.bit_length() - 1
        x ^= lb
        na = adj[a]
        # c candidates: in N(a) and distance-2 from d for some d in N(v)
        # D2[d] for d in N(v): union over d in av of D2[d]
        # precompute union of D2 over av
        uD2 = 0
        y = av
        while y:
            lb2 = y & (-y)
            d = lb2.bit_length() - 1
            uD2 |= D2[d]
            y ^= lb2
        # c must be in N(a) and in uD2, c != a, and c != d for the specific d
        # but we also need b in N(a) & N(c)
        cc = (na & uD2) & ~(1 << a)
        while cc:
            lb3 = cc & (-cc)
            c = lb3.bit_length() - 1
            cc ^= lb3
            if (na & adj[c]) & ~(1 << a) & ~(1 << c):
                return True
    return False

def _edge_count(adj):
    return sum(adj[i].bit_count() for i in range(N)) // 2

def _greedy_order(adj, order):
    """Add all possible edges in given order."""
    for (u, v) in order:
        if not (adj[u] >> v) & 1:
            if not _would_create_c6(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                _recompute_d2(adj)

def _local_search(adj, deadline):
    """Try to add edges greedily in random order repeatedly."""
    best = _edge_count(adj)
    while time.time() < deadline:
        order = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(order)
        _greedy_order(adj, order)
        cnt = _edge_count(adj)
        if cnt > best:
            best = cnt
        if time.time() >= deadline:
            break
    return best

def construct():
    start = time.time()
    deadline = start + 115  # safety margin under 120s
    best_adj = None
    best_cnt = 0

    def seed_adj():
        adj = [0] * N
        _recompute_d2(adj)
        return adj

    # Try many random restarts with random initial greedy fill + local search.
    trials = 0
    while time.time() < deadline:
        trials += 1
        adj = seed_adj()
        # random initial greedy fill
        order = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(order)
        _greedy_order(adj, order)
        _local_search(adj, deadline)
        cnt = _edge_count(adj)
        if cnt > best_cnt:
            best_cnt = cnt
            best_adj = list(adj)
        if time.time() >= deadline:
            break

    # Fallback: at least the known 8xK5 (104 edges) if we somehow did worse.
    if best_adj is None or best_cnt < 104:
        adj = seed_adj()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        best_adj = adj
        best_cnt = _edge_count(adj)

    edges = []
    for u in range(N):
        x = best_adj[u] & ((1 << (u + 1)) - 1)
        while x:
            lb = x & (-x)
            v = lb.bit_length() - 1
            edges.append([u, v])
            x ^= lb
    return edges
