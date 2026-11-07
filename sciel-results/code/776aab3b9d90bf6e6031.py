"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _check_triangle_free(n, adj):
    for u in range(n):
        neigh_u = adj[u]
        for v in neigh_u:
            if v <= u:
                continue
            if neigh_u & adj[v]:
                return False
    return True

def _bfs_diameter(n, adj):
    max_dist = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                    q.append(v)
    return max_dist

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = P['n']
    base_edges = set(tuple(sorted(e)) for e in P['edges'])
    
    # Precompute all non-base edges
    all_non_base = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in base_edges]
    
    best_result = None
    best_count = float('inf')
    
    # Try all combinations of jump lengths
    # A "jump" of length k means adding edges (i, i+k) for all valid i
    # Try single jumps, pairs, and triples
    jump_candidates = list(range(3, n))
    
    def try_jump_set(jumps):
        nonlocal best_result, best_count
        if time.time() > deadline:
            return
        edges = set()
        for jump in jumps:
            for u in range(n - jump):
                edges.add((u, u + jump))
        edge_list = list(edges)
        if len(edge_list) >= best_count:
            return
        adj = _build_adj(n, edge_list)
        if not _check_triangle_free(n, adj):
            return
        diam = _bfs_diameter(n, adj)
        if diam <= 4 and len(edge_list) < best_count:
            best_count = len(edge_list)
            best_result = edge_list[:]
    
    # Try single jumps
    for a in jump_candidates:
        try_jump_set((a,))
    
    # Try pairs
    for i, a in enumerate(jump_candidates):
        for b in jump_candidates[i+1:]:
            try_jump_set((a, b))
    
    # Try triples
    for i, a in enumerate(jump_candidates):
        for j, b in enumerate(jump_candidates[i+1:]):
            for c in jump_candidates[j+1:]:
                try_jump_set((a, b, c))
    
    if best_result is not None:
        return [[int(u), int(v)] for u, v in best_result]
    
    # Fallback
    return [[u, v] for u in range(n) for v in range(u + 1, n) if (u - v) % 2 and (u, v) not in base_edges]
# EVOLVE-BLOCK-END
