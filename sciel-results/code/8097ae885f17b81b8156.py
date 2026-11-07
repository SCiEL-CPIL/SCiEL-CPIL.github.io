import time
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_BASE = P['edges']

def _bipartite_ok(u, v):
    # Path is bipartite with even/odd partition. Adding only even-odd edges keeps it bipartite -> triangle-free.
    return (u ^ v) & 1 == 1

def _build_adj(edge_set):
    adj = [set() for _ in range(_N)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_diameter(adj):
    n = _N
    diam = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = [s]
        head = 0
        maxd = 0
        while head < len(q):
            u = q[head]; head += 1
            du = dist[u]
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    if du + 1 > maxd:
                        maxd = du + 1
                    q.append(v)
        if maxd > diam:
            diam = maxd
    return diam

def _is_connected(adj):
    seen = [False] * _N
    q = [0]
    seen[0] = True
    head = 0
    while head < len(q):
        u = q[head]; head += 1
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return all(seen)

def _valid(added):
    base = set(map(tuple, _BASE))
    es = set()
    for u, v in added:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in base:
            return False
        if key in es:
            return False
        es.add(key)
    # triangle-free guaranteed by bipartite construction; still verify cheaply
    adj = _build_adj(es)
    if not _is_connected(adj):
        return False
    if _bfs_diameter(adj) > 4:
        return False
    return True

def _candidate_pool():
    base = set(map(tuple, _BASE))
    pool = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (u, v) in base:
                continue
            if _bipartite_ok(u, v):
                pool.append((u, v))
    return pool

def _greedy_build(pool, seed_edges, rng):
    # Start from seed_edges (a set of (u,v)), grow until diameter <= 4.
    es = set(seed_edges)
    adj = _build_adj(es)
    # Candidate ordering: prefer edges that connect far-apart vertices / high potential.
    cand = [e for e in pool if e not in es]
    # Precompute base path distance to bias toward long-range edges.
    def path_dist(a, b):
        return abs(a - b)
    # Sort candidates by descending path distance (long-range first) with randomization.
    cand.sort(key=lambda e: (-path_dist(e[0], e[1]), rng.random()))
    for e in cand:
        if _bfs_diameter(adj) <= 4:
            break
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
        es.add(e)
    return es

def _try_remove(es, rng, max_iter=200):
    # Try to remove redundant edges while keeping diameter <= 4 and connected.
    es = set(es)
    improved = True
    it = 0
    while improved and it < max_iter:
        it += 1
        improved = False
        for e in list(es):
            es2 = es - {e}
            adj = _build_adj(es2)
            if _is_connected(adj) and _bfs_diameter(adj) <= 4:
                es = es2
                improved = True
                break
    return es

def _one_attempt(rng):
    pool = _candidate_pool()
    # Random seed edges to diversify: pick a few long-range random edges.
    long_edges = [e for e in pool if abs(e[0] - e[1]) >= 6]
    seed_edges = set()
    k = rng.randint(0, 4)
    for e in rng.sample(long_edges, min(k, len(long_edges))):
        seed_edges.add(e)
    es = _greedy_build(pool, seed_edges, rng)
    es = _try_remove(es, rng)
    return es

def _construct_direct():
    """Directly construct a valid solution with diameter <= 4, triangle-free, minimal edges."""
    # Add edges to create shortcuts. We'll use a greedy approach with verification.
    n = _N
    base = set(map(tuple, _BASE))
    es = set()
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        key = (min(u,v), max(u,v))
        if key in base or key in es:
            return False
        if (u ^ v) & 1 == 0:
            return False  # not bipartite
        es.add(key)
        adj[u].add(v)
        adj[v].add(u)
        return True
    
    def diam():
        if not _is_connected(adj):
            return 100
        return _bfs_diameter(adj)
    
    # Candidate edges sorted by descending path distance
    pool = []
    for u in range(n):
        for v in range(u+1, n):
            if (u,v) in base:
                continue
            if (u ^ v) & 1 == 1:
                pool.append((u,v))
    pool.sort(key=lambda e: -abs(e[0]-e[1]))
    
    for e in pool:
        if diam() <= 4:
            break
        add_edge(e[0], e[1])
    
    # Now try to remove edges
    improved = True
    while improved:
        improved = False
        for e in list(es):
            u, v = e
            es2 = es - {e}
            adj2 = _build_adj(es2)
            if _is_connected(adj2) and _bfs_diameter(adj2) <= 4:
                es = es2
                adj = adj2
                improved = True
                break
    
    return es

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    # Direct construction
    es = _construct_direct()
    if es and _valid(list(es)):
        return [[int(u), int(v)] for (u, v) in sorted(es)]
    # Fallback: brute force greedy
    pool = _candidate_pool()
    es = _greedy_build(pool, set(), random.Random(12345))
    es = _try_remove(es, random.Random(12345))
    if es and _valid(list(es)):
        return [[int(u), int(v)] for (u, v) in sorted(es)]
    return []
# EVOLVE-BLOCK-END
