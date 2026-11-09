import random
import time
import math
import sys

# Increase recursion limit for DFS
sys.setrecursionlimit(10000)

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edges = P['edge_order']
    
    # Map edge to position (1-based)
    edge_pos = {}
    for idx, (u, v) in enumerate(edges, 1):
        if u > v:
            u, v = v, u
        edge_pos[(u, v)] = idx
        edge_pos[(v, u)] = idx # Just in case, though we normalize

    # Sort edges by position (they are already in order, but let's be explicit)
    # We will iterate K from 24 to 276
    
    start_time = time.time()
    time_limit = 55.0
    
    # Pre-compute all possible edges? No, just use the list.
    
    # We need to find the smallest K such that G_K has a Hamiltonian cycle.
    # G_K contains edges with position <= K.
    
    # Helper to check if a graph (adjacency list) has a Hamiltonian cycle using DFS
    # Returns the cycle (list of vertices) or None
    def find_hamiltonian_cycle(adj, n, start_node=0, node_limit=500000):
        # adj: list of sets/lists of neighbors
        # We fix start_node=0 to reduce symmetry
        visited = [False] * n
        path = [start_node]
        visited[start_node] = True
        nodes_visited = 0
        
        def dfs(u):
            nonlocal nodes_visited
            nodes_visited += 1
            if nodes_visited > node_limit:
                return False
            
            if len(path) == n:
                # Check if we can close the cycle
                if start_node in adj[u]:
                    return True
                return False
            
            # Warnsdorff's rule: sort neighbors by their degree in the remaining graph
            # To be fast, we just sort by current degree in adj
            # But we need to consider unvisited neighbors
            candidates = []
            for v in adj[u]:
                if not visited[v]:
                    # Heuristic: prefer v with fewer unvisited neighbors
                    deg = 0
                    for w in adj[v]:
                        if not visited[w]:
                            deg += 1
                    candidates.append((deg, v))
            
            if not candidates:
                return False
            
            candidates.sort()
            
            for deg, v in candidates:
                visited[v] = True
                path.append(v)
                if dfs(v):
                    return True
                path.pop()
                visited[v] = False
            
            return False
        
        if dfs(start_node):
            return path
        return None

    # Iterate K
    # Lower bound for K is N (24)
    for K in range(24, 277):
        if time.time() - start_time > time_limit:
            break
            
        # Build adjacency list for edges with pos <= K
        adj = [set() for _ in range(n)]
        for idx in range(K):
            u, v = edges[idx]
            adj[u].add(v)
            adj[v].add(u)
        
        # Quick Check 1: Degrees
        valid = True
        for i in range(n):
            if len(adj[i]) < 2:
                valid = False
                break
        
        if not valid:
            continue
            
        # Quick Check 2: Connectivity (BFS/DFS from 0)
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 0
        while stack:
            u = stack.pop()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        
        if count < n:
            continue
            
        # Try to find cycle
        cycle = find_hamiltonian_cycle(adj, n)
        if cycle:
            return cycle
            
    # Fallback: Return a random cycle if we run out of time or fail
    # This should not happen if the logic is correct, but just in case
    return list(range(n))
# EVOLVE-BLOCK-END
