import sys
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edge_order = P['edge_order']
    
    # Map edge (u, v) to its arrival index (1-based)
    # Use a dictionary for fast lookup
    edge_index = {}
    for i, edge in enumerate(edge_order):
        u, v = edge
        if u > v:
            u, v = v, u
        edge_index[(u, v)] = i + 1
        
    # Function to check if a Hamilton cycle exists in the subgraph of edges with index <= k
    # Returns the cycle if found, else None
    def find_hamilton_cycle(k):
        # Build adjacency list for edges with index <= k
        adj = defaultdict(list)
        for (u, v), idx in edge_index.items():
            if idx <= k:
                adj[u].append(v)
                adj[v].append(u)
        
        # Check basic connectivity and degree constraints
        # Every node must have degree at least 2
        for i in range(n):
            if len(adj[i]) < 2:
                return None
        
        # Check if graph is connected
        visited = [False] * n
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            u = queue.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        if count != n:
            return None

        # DFS to find Hamilton cycle
        # Start at 0, end at 0
        path = [0]
        visited = [False] * n
        visited[0] = True
        
        def dfs(u, depth):
            if depth == n - 1:
                # Check if last node connects to start
                if 0 in adj[u]:
                    return True
                return False
            
            # Heuristic: sort neighbors by degree (lowest first) to prune faster? 
            # Or just iterate. For small n, simple iteration is fine.
            # Let's try sorting by degree to explore "tight" nodes first
            neighbors = sorted(adj[u], key=lambda x: len(adj[x]))
            
            for v in neighbors:
                if not visited[v]:
                    # Pruning: if v is not the last node, it must have at least 2 unvisited neighbors 
                    # or one unvisited and one visited (the current path's previous? no, cycle)
                    # Actually, if v is not the last node (depth+1 < n-1), 
                    # and v is not connected to 0, it must have at least 2 unvisited neighbors 
                    # available to continue the path and eventually close the cycle.
                    # This is a weak constraint. A stronger one: 
                    # If we visit v, and v is not the last node, 
                    # the number of unvisited neighbors of v (excluding those already in path? no, just unvisited)
                    # must be >= 2 if v is not adjacent to 0? 
                    # Let's stick to a simpler check: 
                    # If depth+1 < n-1, and v is not connected to 0, 
                    # then v must have at least 2 unvisited neighbors.
                    if depth + 1 < n - 1:
                        unvisited_neighbors = [w for w in adj[v] if not visited[w]]
                        if 0 not in adj[v]:
                            if len(unvisited_neighbors) < 2:
                                continue
                        else:
                            # If v connects to 0, it could be the end, but we are not at end yet.
                            # So it still needs to continue.
                            if len(unvisited_neighbors) < 1:
                                continue

                    visited[v] = True
                    path.append(v)
                    if dfs(v, depth + 1):
                        return True
                    path.pop()
                    visited[v] = False
                    
            return False
            
        if dfs(0, 0):
            return path
        return None

    # Binary search for the minimum k
    # Lower bound: n (at least n edges)
    # Upper bound: n * (n-1) / 2
    lo = n
    hi = n * (n - 1) // 2
    best_cycle = None
    best_k = hi + 1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds to be safe
    
    while lo <= hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        cycle = find_hamilton_cycle(mid)
        if cycle:
            best_cycle = cycle
            best_k = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is None:
        # Fallback to seed if no cycle found (should not happen)
        return list(range(n))
        
    return best_cycle
# EVOLVE-BLOCK-END
