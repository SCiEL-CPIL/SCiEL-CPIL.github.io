"""Packing trees T1..T20 into K20 via backtracking with forward checking."""
from random import Random
import time

_DEADLINE = 145.0  # wall-clock safety margin (limit ~150s)


def _instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees


def _build_adjacency(edges, k):
    adj = [[] for _ in range(k)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def construct():
    t0 = time.time()
    N = 20
    pairs = _all_pairs(N)
    pidx = {}
    for a, b in pairs:
        pidx[(a, b)] = pidx[(b, a)] = len(pidx)
    M = len(pairs)  # 190

    trees = _instance_trees(N, 611)
    # per tree: (n, adjacency, list of (u,v,edge_index))
    tinfo = []
    for k in range(1, N + 1):
        edges = trees[k - 1]
        adj = _build_adjacency(edges, k) if k >= 2 else [[] for _ in range(k)]
        elist = []
        for u, v in edges:
            elist.append((u, v, pidx[(u, v)]))
        tinfo.append((k, adj, elist))

    used = bytearray(M)
    result = [None] * N

    # Order: skip k=1 (no edges), place larger trees first (more constrained).
    order = [k for k in range(1, N) if k >= 2]
    order.sort(key=lambda k: -tinfo[k - 1][0])

    # Precompute, per tree, the list of host-vertex assignments to test,
    # and a fast edge-existence check.
    # For a tree with k vertices, we assign host vertices 0..k-1 to tree
    # vertices via a permutation. We do this recursively with MRV on edges.

    def solve():
        # used is global; result filled progressively
        if not order:
            return True
        k = order.pop()
        n, adj, elist = tinfo[k - 1]
        # Try to embed tree of size n into distinct host vertices.
        # We pick an ordering of tree vertices: root at 0, then BFS order.
        # Use backtracking assigning tree vertex -> host vertex, checking
        # edge availability as we go.
        # Build BFS order of tree vertices.
        seen = [False] * n
        bfs = [0]
        seen[0] = True
        for v in bfs:
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    bfs.append(w)
        # parent edge for each vertex (edge connecting it to its parent in BFS)
        # We'll assign host[v] for v in bfs order; when assigning v (not root),
        # we need host[parent] known and edge (host[parent], host[v]) free.
        # Determine parent of each vertex.
        parent = [None] * n
        for v in bfs[1:]:
            for w in adj[v]:
                if seen[w] and (parent[w] is None or w == 0):
                    pass
        # simpler: rebuild parent via BFS
        parent = [None] * n
        seen2 = [False] * n
        seen2[0] = True
        q = [0]
        for v in q:
            for w in adj[v]:
                if not seen2[w]:
                    seen2[w] = True
                    parent[w] = v
                    q.append(w)
        bfs = q  # BFS order
        # For each vertex, the set of candidate host vertices.
        host = [-1] * n
        host_used = bytearray(N)

        def rec(pos):
            if time.time() - t0 > _DEADLINE:
                return False
            if pos == n:
                result[k - 1] = list(host)
                return True
            v = bfs[pos]
            # candidate host vertices: not used; if v has parent, edge must be free
            pv = parent[v]
            if pv is None:
                for h in range(N):
                    if not host_used[h]:
                        host[v] = h
                        host_used[h] = 1
                        if rec(pos + 1):
                            return True
                        host_used[h] = 0
                        host[v] = -1
            else:
                ph = host[pv]
                for h in range(N):
                    if not host_used[h] and not used[pidx[(ph, h)]]:
                        host[v] = h
                        host_used[h] = 1
                        if rec(pos + 1):
                            return True
                        host_used[h] = 0
                        host[v] = -1
            return False

        if rec(0):
            # mark edges used
            for u, v, ei in elist:
                used[ei] = 1
            return True
        order.append(k)
        return False

    ok = solve()
    # value = sum of edges of embedded trees
    value = 0
    for k in range(1, N):
        if result[k - 1] is not None:
            value += (k - 1)
    return result


if __name__ == "__main__":
    r = construct()
    print("embedded:", sum(1 for x in r if x is not None), "value:", sum((i) for i, x in enumerate(r) if x is not None))
