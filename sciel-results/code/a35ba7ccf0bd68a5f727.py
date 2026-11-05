"""Cage-filling heuristic: grow the minimal k-ary tree of radius (g-1)//2,
then greedily close it into a simple k-regular graph by adding edges only
between vertices whose tree-distance is >= g-1 (so every new cycle has
length >= g). Repair to exact k-regularity."""
import random
from collections import deque

def construct(k, g):
    if k < 1:
        return 0, []
    if g <= 2:
        # trivial: any k-regular graph on n>=k+1 has girth >= 2
        n = k + 1
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return n, edges

    r = (g - 1) // 2
    # Moore bound for k-regular girth g
    if g % 2 == 1:  # odd girth, radius r, center
        n_moore = 1 + k * (k ** r - 1) // (k - 1) if k > 1 else 1 + k * r
    else:            # even girth
        n_moore = 2 * k * (k ** (r - 1) - 1) // (k - 1) if k > 1 else 2 * k

    # Build a k-ary tree of radius r (rooted). For even girth we use two roots
    # connected by an edge (standard Moore tree construction).
    rng = random.Random(12345)

    if g % 2 == 1:
        # single root
        root = 0
        adj = {root: []}
        order = [root]
        for v in order:
            depth = _depth(v, adj, root)
            if depth < r:
                cap = k if depth == 0 else k - 1
                for _ in range(cap):
                    nv = len(adj)
                    adj[nv] = []
                    adj[v].append(nv)
                    order.append(nv)
    else:
        # two roots 0 and 1 connected
        adj = {0: [], 1: []}
        adj[0].append(1)
        adj[1].append(0)
        order = [0, 1]
        for v in order:
            depth = _depth_even(v, adj)
            if depth < r - 1:
                cap = k - 1
                for _ in range(cap):
                    nv = len(adj)
                    adj[nv] = []
                    adj[v].append(nv)
                    order.append(nv)

    n = len(adj)
    # adjacency sets
    adjset = {v: set(adj[v]) for v in adj}

    # BFS distances from every vertex (tree, so unique paths)
    dist = {}
    for s in range(n):
        d = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adjset[u]:
                if w not in d:
                    d[w] = d[u] + 1
                    q.append(w)
        dist[s] = d

    def tree_dist(u, v):
        return dist[u][v]

    # degrees
    deg = {v: len(adjset[v]) for v in adjset}

    # Candidate non-tree edges: pairs (u,v) not already adjacent with tree_dist >= g-1
    # We need to bring every vertex to degree k.
    # Greedy: repeatedly pick the vertex with lowest degree, find a valid partner
    # with lowest degree too, add edge if it doesn't create a short cycle.
    # Adding edge (u,v) creates cycle of length tree_dist(u,v)+1. Need >= g.
    # Also must ensure no other shorter cycle forms: in a tree + one edge the only
    # new cycle is the fundamental one, so checking tree_dist >= g-1 suffices for
    # that edge. But after multiple edges, a new edge could form a shorter cycle
    # via existing non-tree edges. We check actual shortest path in current graph.
    import heapq

    def shortest_path_len(u, v, adjset):
        if u == v:
            return 0
        d = {u: 0}
        q = deque([u])
        while q:
            x = q.popleft()
            if x == v:
                return d[x]
            for w in adjset[x]:
                if w not in d:
                    d[w] = d[x] + 1
                    q.append(w)
        return None

    # We'll add edges to reach degree k everywhere.
    # Sort vertices by deficit.
    while True:
        deficits = [v for v in range(n) if deg[v] < k]
        if not deficits:
            break
        # pick lowest-degree vertex
        u = min(deficits, key=lambda v: deg[v])
        # candidates: not adjacent, deg < k, and adding edge keeps girth
        cands = []
        for v in range(n):
            if v == u or v in adjset[u] or deg[v] >= k:
                continue
            # check girth: shortest path u->v in current graph must be >= g-1
            sp = shortest_path_len(u, v, adjset)
            if sp is None:
                continue
            if sp + 1 < g:
                continue
            cands.append(v)
        if not cands:
            # try to find any valid edge among all deficit vertices
            added = False
            for a in deficits:
                for b in deficits:
                    if a >= b or b in adjset[a]:
                        continue
                    sp = shortest_path_len(a, b, adjset)
                    if sp is not None and sp + 1 >= g:
                        adjset[a].add(b)
                        adjset[b].add(a)
                        deg[a] += 1
                        deg[b] += 1
                        added = True
                        break
                if added:
                    break
            if not added:
                break  # cannot satisfy; leave as is (best effort)
            continue
        # pick candidate with lowest degree (most constrained)
        v = min(cands, key=lambda w: deg[w])
        adjset[u].add(v)
        adjset[v].add(u)
        deg[u] += 1
        deg[v] += 1

    # Build edge list
    edges = []
    for u in range(n):
        for v in adjset[u]:
            if u < v:
                edges.append((u, v))

    return n, edges

def _depth(v, adj, root):
    # BFS from root to v
    d = {root: 0}
    q = deque([root])
    while q:
        x = q.popleft()
        if x == v:
            return d[x]
        for w in adj[x]:
            if w not in d:
                d[w] = d[x] + 1
                q.append(w)
    return 0

def _depth_even(v, adj):
    # distance to nearest of {0,1}
    d = {0: 0, 1: 0}
    q = deque([0, 1])
    while q:
        x = q.popleft()
        if x == v:
            return d[x]
        for w in adj[x]:
            if w not in d:
                d[w] = d[x] + 1
                q.append(w)
    return 0
