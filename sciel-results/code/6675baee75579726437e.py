"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_graph(added_edges):
    """Build adjacency from base path + added edges."""
    n = 24
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_diameter(adj):
    """Compute diameter using BFS from all vertices."""
    n = len(adj)
    max_dist = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return -1  # disconnected
        max_dist = max(max_dist, max(dist))
    return max_dist

def _is_triangle_free(adj):
    """Check if graph is triangle-free."""
    n = len(adj)
    adj_set = [set(a) for a in adj]
    for u in range(n):
        for i, v in enumerate(adj[u]):
            for w in adj[u][i+1:]:
                if w in adj_set[v]:
                    return False
    return True

def _is_valid(added_edges):
    """Check if added edges produce a valid graph."""
    # Check for duplicate edges
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
    for u, v in added_edges:
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    adj = _build_graph(added_edges)
    if not _is_triangle_free(adj):
        return False
    diameter = _bfs_diameter(adj)
    if diameter < 0 or diameter > 4:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin
    
    # Known optimal candidate: 10 edges
    # Hub 12 connects to specific odd vertices to bridge the path efficiently
    base_solution = [
        [12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
        [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]
    ]
    
    # Verify base solution
    if not _is_valid(base_solution):
        # Fallback to empty if base is somehow invalid (should not happen)
        return []
    
    best_solution = base_solution
    
    # Try to reduce to 9 edges by removing one edge from the 10-edge solution
    # This is a cheap local search (only 10 checks)
    for i in range(len(base_solution)):
        if time.time() > deadline:
            break
        candidate = base_solution[:i] + base_solution[i+1:]
        if _is_valid(candidate):
            best_solution = candidate
            break  # Found a 9-edge solution, which is better than 10
    
    # Ensure return type is correct
    result = []
    for e in best_solution:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
