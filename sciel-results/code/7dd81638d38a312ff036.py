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

def _check_solution_fast(n, base_edges, added_edges, max_diameter=4):
    """Fast check: triangle-free, connected, diameter <= max_diameter."""
    all_edges = list(base_edges) + list(added_edges)
    
    # Check for duplicate edges
    edge_set = set()
    for u, v in all_edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Build adjacency with sets for fast intersection
    adj = [set() for _ in range(n)]
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    
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
    
    # Check triangle-free: for each edge (u,v), no common neighbor
    for u, v in all_edges:
        if adj[u] & adj[v]:
            return False
    
    # Check diameter via BFS
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_diameter:
                        return False
                    q.append(v)
        if -1 in dist:
            return False
    
    return True

def _get_max_diameter(n, all_edges):
    """Get the maximum distance in the graph."""
    adj = [set() for _ in range(n)]
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    max_d = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_d:
                        max_d = dist[v]
                    q.append(v)
    return max_d

def _is_valid_edge(n, all_edges, u, v):
    """Check if adding edge (u,v) keeps the graph triangle-free and non-duplicate."""
    if u == v:
        return False
    key = (min(u, v), max(u, v))
    for a, b in all_edges:
        if (min(a, b), max(a, b)) == key:
            return False
    
    # Check no common neighbor (triangle-free)
    adj = [set() for _ in range(n)]
    for a, b in all_edges:
        adj[a].add(b)
        adj[b].add(a)
    
    if adj[u] & adj[v]:
        return False
    
    return True

def _local_search_aggressive(n, base_edges, added_edges, max_diameter=4, deadline=None):
    """Aggressive local search: remove and swap edges to minimize count."""
    current_added = [list(e) for e in added_edges]
    best_added = [list(e) for e in current_added]
    
    # Phase 1: Remove edges
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(current_added)):
            if deadline and time.time() > deadline:
                break
            test_added = current_added[:i] + current_added[i+1:]
            if _check_solution_fast(n, base_edges, test_added, max_diameter):
                current_added = test_added
                improved = True
                break
    
    # Phase 2: Try swaps (remove one, add another)
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(current_added)):
            if deadline and time.time() > deadline:
                break
            # Try removing edge i and adding a different edge
            without_i = current_added[:i] + current_added[i+1:]
            all_edges_no_i = list(base_edges) + without_i
            adj_no_i = [set() for _ in range(n)]
            for a, b in all_edges_no_i:
                adj_no_i[a].add(b)
                adj_no_i[b].add(a)
            
            # Find candidate edges to add
            for u in range(n):
                if deadline and time.time() > deadline:
                    break
                for v in range(u+1, n):
                    if deadline and time.time() > deadline:
                        break
                    key = (min(u, v), max(u, v))
                    # Skip if already in graph
                    if key in set((min(a,b), max(a,b)) for a, b in all_edges_no_i):
                        continue
                    # Check triangle-free
                    if adj_no_i[u] & adj_no_i[v]:
                        continue
                    
                    test_added = without_i + [[u, v]]
                    if _check_solution_fast(n, base_edges, test_added, max_diameter):
                        # Same number of edges but different structure - skip for now
                        # We want to reduce count, so only accept if we can then remove
                        pass
            # Only accept removals that lead to valid solutions
            if _check_solution_fast(n, base_edges, without_i, max_diameter):
                current_added = without_i
                improved = True
                break
    
    # Phase 3: Final removal pass
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(current_added)):
            if deadline and time.time() > deadline:
                break
            test_added = current_added[:i] + current_added[i+1:]
            if _check_solution_fast(n, base_edges, test_added, max_diameter):
                current_added = test_added
                improved = True
                break
    
    return current_added

def construct():
    start = time.time()
    deadline = start + 80  # Safety margin before 90s limit
    n = P['n']
    base_edges = P['edges']
    max_diameter = P['diameter']
    
    # Jump-based initial guess: connect distant pairs
    # Strategy: use edges that span large distances to quickly reduce diameter
    # For a path of 24 vertices, we need to connect vertices that are far apart
    
    # Try multiple initial strategies and pick the best
    strategies = []
    
    # Strategy 1: Hub at center (12)
    hub1 = 12
    s1 = []
    for node in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]:
        s1.append([hub1, node])
    strategies.append(s1)
    
    # Strategy 2: Hub at 11 (slightly off-center)
    hub2 = 11
    s2 = []
    for node in [0, 2, 4, 6, 8, 13, 15, 17, 19, 21, 23]:
        s2.append([hub2, node])
    strategies.append(s2)
    
    # Strategy 3: Jump edges - connect pairs with large distance
    s3 = []
    # Connect 0-12, 1-13, 2-14, 3-15, 4-16, 5-17, 6-18, 7-19, 8-20, 9-21, 10-22, 11-23
    # But need to check triangle-free
    jump_pairs = [(0, 12), (1, 13), (2, 14), (3, 15), (4, 16), (5, 17), (6, 18), (7, 19), (8, 20), (9, 21), (10, 22), (11, 23)]
    for u, v in jump_pairs:
        if u % 2 != v % 2:  # Bipartite check
            s3.append([u, v])
    strategies.append(s3)
    
    # Strategy 4: Fewer hub edges
    hub4 = 12
    s4 = []
    for node in [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]:
        s4.append([hub4, node])
    strategies.append(s4)
    
    # Strategy 5: Two hubs
    s5 = []
    for node in [1, 3, 5, 7, 9]:
        s5.append([11, node])
    for node in [15, 17, 19, 21, 23]:
        s5.append([13, node])
    s5.append([11, 13])
    strategies.append(s5)
    
    # Evaluate all strategies and pick the best (fewest edges after local search)
    best_result = None
    best_count = float('inf')
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        
        # Verify the strategy is valid
        if not _check_solution_fast(n, base_edges, strategy, max_diameter):
            continue
        
        # Apply local search
        result = _local_search_aggressive(n, base_edges, strategy, max_diameter, deadline)
        
        if _check_solution_fast(n, base_edges, result, max_diameter):
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # If no valid strategy found, fallback to greedy construction
    if best_result is None:
        # Build from scratch using greedy
        added = []
        all_edges = list(base_edges)
        
        for _ in range(50):
            if time.time() > deadline:
                break
            
            max_d = _get_max_diameter(n, all_edges)
            if max_d <= max_diameter:
                break
            
            # Find best edge to add
            best_edge = None
            best_max_d = max_d
            
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u+1, n):
                    if time.time() > deadline:
                        break
                    if not _is_valid_edge(n, all_edges, u, v):
                        continue
                    
                    test_edges = all_edges + [[u, v]]
                    test_max_d = _get_max_diameter(n, test_edges)
                    
                    if test_max_d < best_max_d:
                        best_max_d = test_max_d
                        best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            added.append([u, v])
            all_edges = all_edges + [[u, v]]
        
        best_result = added
    
    # Final verification
    if best_result is None or not _check_solution_fast(n, base_edges, best_result, max_diameter):
        # Emergency fallback
        best_result = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9], [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
    
    return [[int(u), int(v)] for u, v in best_result]
# EVOLVE-BLOCK-END
