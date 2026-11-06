import time
import sys
sys.setrecursionlimit(10000)

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = 24
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Build adjacency with arrival indices
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(P['edge_order'], 1):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    # Precompute: for each vertex, list of (neighbor, arrival_index) sorted by arrival_index
    for u in range(n):
        adj[u].sort(key=lambda x: x[1])
    
    def has_hamilton_cycle(K):
        """Check if there's a Hamilton cycle using only edges with arrival index <= K."""
        # Build adjacency for threshold K
        adj_K = [[] for _ in range(n)]
        for u in range(n):
            for v, idx in adj[u]:
                if idx <= K:
                    adj_K[u].append(v)
                else:
                    break  # Since sorted by idx
        
        # Quick check: every vertex must have degree >= 2
        for u in range(n):
            if len(adj_K[u]) < 2:
                return None
        
        # Build set for fast lookup
        adj_K_set = [set(adj_K[u]) for u in range(n)]
        
        # DFS with pruning
        start = 0
        neighbors_of_start = sorted(adj_K[start])
        
        # To reduce symmetry, fix the second vertex to be the smallest neighbor
        # Actually, let's try all but with a time limit
        for second in neighbors_of_start:
            path = [start, second]
            visited = (1 << start) | (1 << second)
            remaining = n - 2
            
            def dfs(current, path, visited, remaining):
                if time.time() > deadline:
                    return None
                
                if remaining == 0:
                    if start in adj_K_set[current]:
                        return path[:]
                    return None
                
                # Pruning 1: check unvisited vertices have degree >= 1 to unvisited set
                # (they need to be in the middle of the remaining path segment)
                unvisited = []
                for v in range(n):
                    if not (visited >> v) & 1 and v != start:
                        unvisited.append(v)
                
                if not unvisited:
                    return None
                
                # Each unvisited vertex must have at least one neighbor in unvisited set
                # OR be connectable as the last vertex (to current, then current to start)
                for v in unvisited:
                    has_unvisited_nb = False
                    for nb in adj_K[v]:
                        if not (visited >> nb) & 1 and nb != start and nb != v:
                            has_unvisited_nb = True
                            break
                    if not has_unvisited_nb:
                        # v can only be last: must connect to current, and current must connect to start
                        # But we don't know the order, so this is a necessary condition
                        # v must connect to at least one vertex that could be before it
                        # Simple: v must have degree >= 1 to (unvisited ∪ {current})
                        # which is always true if has_unvisited_nb is false but v connects to current
                        # Actually if has_unvisited_nb is false, v has no unvisited neighbors
                        # So v must be the last vertex, connecting to current
                        # This means current must connect to v, and v must connect to start
                        # But we're not at the last step yet, so this is a problem
                        # Unless remaining == 1
                        if remaining > 1:
                            return None
                        # If remaining == 1, v is the last, must connect to current and start
                        if current not in adj_K_set[v] or start not in adj_K_set[v]:
                            return None
                
                # Pruning 2: connectivity of unvisited subgraph
                # The unvisited vertices must form a connected subgraph (or be connectable through current)
                # Simple BFS/DFS check
                if len(unvisited) > 1:
                    # Check connectivity of unvisited subgraph
                    seen = {unvisited[0]}
                    stack = [unvisited[0]]
                    while stack:
                        x = stack.pop()
                        for nb in adj_K[x]:
                            if nb in unvisited and nb not in seen:
                                seen.add(nb)
                                stack.append(nb)
                    if len(seen) != len(unvisited):
                        # Disconnected, but they could be connected through current
                        # Check if each component has an edge to current
                        # For simplicity, just check if current connects to at least one in each component
                        # This is complex, so let's just fail if disconnected
                        # Actually, they could be connected through current (current is in the path)
                        # So the unvisited vertices don't need to be connected to each other,
                        # they just need to be reachable from current in the remaining path
                        # This is always true if each has degree >= 1 to (unvisited ∪ {current})
                        # Which we already checked. So skip this pruning.
                        pass
                
                # Choose next vertex: try neighbors of current that are unvisited
                candidates = []
                for nb in adj_K[current]:
                    if not (visited >> nb) & 1 and nb != start:
                        candidates.append(nb)
                
                # Sort candidates by degree (fewest options first) for better pruning
                candidates.sort(key=lambda v: len(adj_K[v]))
                
                for nb in candidates:
                    new_visited = visited | (1 << nb)
                    path.append(nb)
                    result = dfs(nb, path, new_visited, remaining - 1)
                    path.pop()
                    if result:
                        return result
                
                return None
            
            result = dfs(second, path, visited, remaining)
            if result:
                return result
        
        return None
    
    # Binary search on K
    lo, hi = 1, 276
    best_K = 276
    best_cycle = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        cycle = has_hamilton_cycle(mid)
        if cycle:
            best_K = mid
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is None:
        # Fallback: try a greedy approach
        # Just return identity
        return list(range(n))
    
    return best_cycle
