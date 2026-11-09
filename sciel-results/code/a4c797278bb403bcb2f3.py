"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_distances(n, adj):
    """Compute all-pairs shortest path distances using BFS from each vertex."""
    dist = [[-1] * n for _ in range(n)]
    for src in range(n):
        dist[src][src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[src][v] == -1:
                    dist[src][v] = dist[src][u] + 1
                    q.append(v)
    return dist

def _check_solution(n, base_edges, added_edges, max_diameter=4):
    """Check if the union of base and added edges is triangle-free, connected, and has diameter <= max_diameter."""
    all_edges = list(base_edges) + list(added_edges)
    
    # Check for duplicate edges
    edge_set = set()
    for u, v in all_edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Build adjacency
    adj = _build_adj(n, all_edges)
    
    # Check connectedness
    visited = [False] * n
    visited[0] = True
    q = deque([0])
    count = 1
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                q.append(v)
    if count != n:
        return False
    
    # Check triangle-free: for each edge (u,v), check no common neighbor
    for u, v in all_edges:
        neighbors_u = set(adj[u])
        neighbors_v = set(adj[v])
        common = neighbors_u & neighbors_v
        if common:
            return False
    
    # Check diameter
    dist = _bfs_distances(n, adj)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1 or dist[i][j] > max_diameter:
                return False
    
    return True

def _greedy_add_edges(n, base_edges, added_edges, max_diameter=4):
    """Try to add edges one at a time to reduce diameter."""
    all_edges = list(base_edges) + list(added_edges)
    adj = _build_adj(n, all_edges)
    dist = _bfs_distances(n, adj)
    
    # Find pairs with distance > max_diameter
    far_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if dist[i][j] > max_diameter:
                far_pairs.append((i, j))
    
    if not far_pairs:
        return added_edges, True
    
    # For each far pair, find candidate edges that could help
    # An edge (u,v) helps if it reduces the distance between some far pairs
    # Simple approach: try adding each possible edge and see if it helps
    
    possible_edges = []
    for i in range(n):
        for j in range(i+1, n):
            key = (min(i, j), max(i, j))
            if key not in set((min(u,v), max(u,v)) for u, v in all_edges):
                # Check triangle-free constraint: i and j must not share a neighbor
                # and the edge must be between different parity (bipartite)
                if i % 2 == j % 2:
                    continue
                # Check no common neighbor
                adj_i = set(_build_adj(n, all_edges)[i])
                adj_j = set(_build_adj(n, all_edges)[j])
                if adj_i & adj_j:
                    continue
                possible_edges.append((i, j))
    
    # Greedily add edges that reduce the maximum distance the most
    current_added = list(added_edges)
    current_all = list(base_edges) + current_added
    
    for _ in range(50):  # Limit iterations
        adj = _build_adj(n, current_all)
        dist = _bfs_distances(n, adj)
        
        # Check if we're done
        max_d = 0
        for i in range(n):
            for j in range(n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
        
        if max_d <= max_diameter:
            return current_added, True
        
        # Find best edge to add
        best_edge = None
        best_score = -1
        
        for u, v in possible_edges:
            # Check if edge already added
            key = (min(u, v), max(u, v))
            if key in set((min(x[0], x[1]), max(x[0], x[1])) for x in current_added):
                continue
            
            # Test adding this edge
            test_edges = current_all + [[u, v]]
            if not _check_solution(n, base_edges, current_added + [[u, v]], max_diameter):
                continue
            
            test_adj = _build_adj(n, test_edges)
            test_dist = _bfs_distances(n, test_adj)
            
            # Score: reduction in maximum distance
            test_max_d = 0
            for i in range(n):
                for j in range(n):
                    if test_dist[i][j] > test_max_d:
                        test_max_d = test_dist[i][j]
            
            score = max_d - test_max_d
            if score > best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        current_added.append(list(best_edge))
    
    return current_added, False

def _local_search_remove(n, base_edges, added_edges, max_diameter=4):
    """Try to remove edges one at a time while maintaining validity."""
    current_added = list(added_edges)
    improved = True
    
    while improved:
        improved = False
        for i in range(len(current_added)):
            # Try removing edge i
            test_added = current_added[:i] + current_added[i+1:]
            if _check_solution(n, base_edges, test_added, max_diameter):
                current_added = test_added
                improved = True
                break
    
    return current_added

def construct():
    start = time.time()
    n = P['n']
    base_edges = P['edges']
    max_diameter = P['diameter']
    
    # Initial solution: hub-and-spoke
    center = 12
    left_spokes = [1, 3, 5, 7, 9]
    right_spokes = [15, 17, 19, 21, 23]
    
    initial_added = []
    for node in left_spokes:
        initial_added.append([center, node])
    for node in right_spokes:
        initial_added.append([center, node])
    
    # Verify initial solution
    if not _check_solution(n, base_edges, initial_added, max_diameter):
        # Fallback: try to build from scratch
        initial_added = []
    
    # Greedy addition if needed
    added, done = _greedy_add_edges(n, base_edges, initial_added, max_diameter)
    
    if not done:
        # Try more aggressive search
        added = initial_added
    
    # Local search: remove unnecessary edges
    final_added = _local_search_remove(n, base_edges, added, max_diameter)
    
    # Verify final solution
    if not _check_solution(n, base_edges, final_added, max_diameter):
        # Fallback to initial
        final_added = initial_added
    
    # Safety margin
    if time.time() - start > 80:
        return [[int(u), int(v)] for u, v in final_added]
    
    return [[int(u), int(v)] for u, v in final_added]
# EVOLVE-BLOCK-END
