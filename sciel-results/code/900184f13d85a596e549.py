"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(added_edges):
    """Build adjacency list from base path + added edges."""
    n = P['n']
    adj = [set() for _ in range(n)]
    for e in P['edges']:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    for e in added_edges:
        u, v = int(e[0]), int(e[1])
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _check_validity(added_edges):
    """Check if added edges produce a valid triangle-free, connected, diameter<=4 graph."""
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    all_edges = set(base_edges)
    for e in added_edges:
        u, v = int(min(e[0], e[1])), int(max(e[0], e[1]))
        if (u, v) in all_edges:
            return False
        all_edges.add((u, v))
    
    adj = [set() for _ in range(n)]
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Triangle-free check: for each vertex, no two neighbors are adjacent
    for u in range(n):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in adj[neighbors[i]]:
                    return False
    
    # Connected check via BFS
    visited = [False] * n
    visited[0] = True
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)
    if not all(visited):
        return False
    
    # Diameter <= 4 check via BFS from each node
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if dist[node] >= 4:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        if any(d == -1 for d in dist):
            return False
    
    return True

def _bipartite_greedy_reduce(candidates, base_adj):
    """
    Start with all candidate even-odd edges, greedily remove edges while maintaining validity.
    candidates: list of (u, v) where u is even, v is odd (or vice versa)
    Returns the reduced set as a list.
    """
    # Start with all candidates
    current = set(candidates)
    
    # Precompute base adjacency for quick checks
    n = P['n']
    
    # Greedy removal: repeatedly try to remove each edge
    improved = True
    while improved:
        improved = False
        for edge in list(current):
            test_set = current - {edge}
            test_list = [list(e) for e in test_set]
            if _check_validity(test_list):
                current = test_set
                improved = True
                break
    
    return [list(e) for e in current]

def _smart_greedy_build(deadline):
    """
    Build a minimal edge set using a more targeted approach:
    1. Identify critical pairs that need shortcutting
    2. Use local search to minimize edges
    """
    n = P['n']
    even_verts = list(range(0, n, 2))
    odd_verts = list(range(1, n, 2))
    
    # Generate all possible even-odd edges (excluding path edges)
    path_edges = set()
    for e in P['edges']:
        path_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    all_candidates = []
    for u in even_verts:
        for v in odd_verts:
            pair = (min(u, v), max(u, v))
            if pair not in path_edges:
                all_candidates.append(pair)
    
    # Strategy: Start with a hub-based solution, then optimize
    # Try multiple hub configurations and pick the best, then reduce
    
    best_result = None
    best_count = float('inf')
    
    # Try hub at each even vertex
    for hub in even_verts:
        if time.time() > deadline:
            break
        
        # Connect hub to all odd vertices not already connected via path
        hub_edges = []
        for v in odd_verts:
            pair = (min(hub, v), max(hub, v))
            if pair not in path_edges:
                hub_edges.append(pair)
        
        if len(hub_edges) < best_count and time.time() < deadline:
            # Try to reduce this hub solution
            reduced = _bipartite_greedy_reduce(hub_edges, None)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_result = reduced
    
    # Also try a global greedy from all candidates
    if time.time() < deadline:
        reduced_all = _bipartite_greedy_reduce(all_candidates, None)
        if len(reduced_all) < best_count:
            best_count = len(reduced_all)
            best_result = reduced_all
    
    if best_result is None:
        # Fallback: hub at 12
        hub = 12
        hub_edges = []
        for v in odd_verts:
            pair = (min(hub, v), max(hub, v))
            if pair not in path_edges:
                hub_edges.append(pair)
        best_result = [list(e) for e in hub_edges]
    
    # Final local optimization: try swapping edges
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges = set(tuple(e) for e in best_result)
        
        # Try removing each edge
        for edge in list(current_edges):
            if time.time() > deadline:
                break
            test_set = current_edges - {edge}
            test_list = [list(e) for e in test_set]
            if _check_validity(test_list):
                best_result = test_list
                improved = True
                break
        
        # Try adding one edge and removing another (1-for-1 swap)
        if not improved and time.time() < deadline:
            for add_edge in all_candidates:
                if add_edge in current_edges:
                    continue
                for rem_edge in list(current_edges):
                    if time.time() > deadline:
                        break
                    test_set = (current_edges - {rem_edge}) | {add_edge}
                    test_list = [list(e) for e in test_set]
                    if _check_validity(test_list):
                        # Only accept if it helps (same count, but might enable further reductions)
                        # Actually, 1-for-1 keeps same count; skip unless we can then remove more
                        pass
    
    return [list(e) for e in best_result]

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill

    n = 24
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))

    even_verts = list(range(0, n, 2))
    odd_verts = list(range(1, n, 2))
    
    # Phase 1: Try smart greedy build
    result = _smart_greedy_build(deadline)
    
    # Ensure proper types: integers must be Python int, lists must be list
    final_result = []
    for e in result:
        u, v = int(e[0]), int(e[1])
        final_result.append([u, v])
    
    return final_result
# EVOLVE-BLOCK-END
