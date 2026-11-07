"""Triangle-free diameter-four augmentation."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_all_pairs(n, adj):
    """Compute all-pairs shortest paths. Returns dist matrix and diameter."""
    dist = [[-1] * n for _ in range(n)]
    for src in range(n):
        dist[src][src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            du = dist[src][u]
            for v in range(n):
                if adj[u][v] and dist[src][v] == -1:
                    dist[src][v] = du + 1
                    q.append(v)
    diam = 0
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1:
                return None, -1  # disconnected
            if dist[i][j] > diam:
                diam = dist[i][j]
    return dist, diam

def _is_valid(n, added_edges, base_adj):
    """Full validation: no duplicates, triangle-free, connected, diameter <= 4."""
    adj = [row[:] for row in base_adj]
    for u, v in added_edges:
        if u == v or adj[u][v]:
            return False
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                return False
        adj[u][v] = True
        adj[v][u] = True
    dist, diam = _bfs_all_pairs(n, adj)
    if dist is None:
        return False
    return diam <= 4

def _can_add_edge(adj, u, v, n):
    """Check if edge (u,v) can be added without creating a triangle."""
    if u == v or adj[u][v]:
        return False
    for w in range(n):
        if w != u and w != v and adj[u][w] and adj[v][w]:
            return False
    return True

def construct():
    # Return a known valid solution with 8 edges
    # This solution is triangle-free, connected, and has diameter <= 4
    # Edges: (12, 1), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23)
    return [[12, 1], [12, 3], [12, 6], [12, 9], [12, 15], [12, 18], [12, 21], [12, 23]]

# EVOLVE-BLOCK-END
