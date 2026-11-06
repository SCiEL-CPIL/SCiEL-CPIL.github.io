import sys
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edges = P['edge_order']
    num_edges = len(edges)
    
    # Map edge to its 1-based index
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        if u > v:
            u, v = v, u
        edge_to_idx[(u, v)] = i + 1

    # Precompute all edges for quick access if needed, but we work with prefixes
    
    def find_hamilton_cycle(K):
        """
        Finds a Hamilton cycle in the graph formed by the first K edges.
        Returns the cycle as a list of vertices [v0, v1, ..., v23] where 
        edges are (v0,v1), ..., (v22,v23), (v23,v0).
        Returns None if no such cycle exists.
        """
        # Build adjacency list for G_K
        adj = [set() for _ in range(n)]
        for i in range(K):
            u, v = edges[i]
            adj[u].add(v)
            adj[v].add(u)
            
        # Quick checks
        for i in range(n):
            if len(adj[i]) < 2:
                return None
                
        # Connectivity check
        visited = [False] * n
        visited[0] = True
        stack = [0]
        count = 0
        while stack:
            node = stack.pop()
            count += 1
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if count != n:
            return None

        # Backtracking with Warnsdorff's heuristic
        # We fix the start vertex to 0 to reduce symmetry
        # path[0] = 0
        path = [0]
        used = [False] * n
        used[0] = True
        
        # Heuristic: sort neighbors by degree of unvisited neighbors
        # This needs to be dynamic.
        
        def backtrack(current):
            if len(path) == n:
                # Check if we can close the cycle
                if 0 in adj[current]:
                    return True
                return False
            
            # Get neighbors of current
            neighbors = list(adj[current])
            # Filter to unvisited
            unvisited_neighbors = [nb for nb in neighbors if not used[nb]]
            
            if not unvisited_neighbors:
                return False
            
            # Warnsdorff's heuristic: sort by number of unvisited neighbors of the neighbor
            # This helps find a solution faster or fail faster
            unvisited_neighbors.sort(key=lambda x: sum(1 for y in adj[x] if not used[y]))
            
            for nb in unvisited_neighbors:
                # Pruning: If nb is the last vertex to be added, it must connect to 0
                # We handle this in the base case, but we can prune earlier.
                # If len(path) + 1 == n - 1 (i.e., adding nb makes path length n-1),
                # then the next vertex (last one) must connect to 0.
                # Actually, simpler: just let the base case handle it.
                
                path.append(nb)
                used[nb] = True
                
                if backtrack(nb):
                    return True
                
                path.pop()
                used[nb] = False
                
            return False
        
        if backtrack(0):
            return path
        return None

    # Binary search for minimal K
    # K must be at least n (24)
    # K is at most num_edges (276)
    low = n
    high = num_edges
    best_cycle = None
    best_K = num_edges + 1
    
    # We want to find the smallest K such that a cycle exists.
    # If a cycle exists for K, it exists for K+1.
    # So we binary search for the first True.
    
    # To be safe against timeouts, we can add a time check, 
    # but for N=24, the backtracking should be fast enough for ~8 iterations.
    
    while low <= high:
        mid = (low + high) // 2
        cycle = find_hamilton_cycle(mid)
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            high = mid - 1
        else:
            low = mid + 1
            
    if best_cycle is None:
        # Fallback to identity if something went wrong (should not happen for K=276)
        return list(range(n))
        
    return best_cycle

# EVOLVE-BLOCK-END
