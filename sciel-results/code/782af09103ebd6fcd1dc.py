import sys
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # We need to find up to 3 edge-disjoint Hamiltonian cycles.
    # We will maintain a set of used edges to ensure disjointness.
    used_edges = set()
    cycles = []
    
    # Helper to check if an edge is available
    def is_edge_available(u, v):
        if u > v:
            u, v = v, u
        return (u, v) not in used_edges

    # DFS to find a Hamiltonian cycle
    # path: list of vertices in current path
    # visited: set of visited vertices
    def find_ham_cycle():
        # Start at 0
        path = [0]
        visited = {0}
        
        # Use recursion with backtracking
        # To avoid recursion depth issues (though 21 is small), 
        # we can use standard recursion as sys.setrecursionlimit is likely sufficient.
        
        def dfs(current, path, visited):
            if len(path) == n:
                # Check if we can close the cycle back to start (0)
                if is_edge_available(current, 0):
                    return list(path)
                return None
            
            # Try neighbors
            # Heuristic: try neighbors with fewer available options first? 
            # For now, standard order.
            for neighbor in adj[current]:
                if neighbor not in visited:
                    if is_edge_available(current, neighbor):
                        visited.add(neighbor)
                        path.append(neighbor)
                        
                        result = dfs(neighbor, path, visited)
                        if result is not None:
                            return result
                            
                        path.pop()
                        visited.remove(neighbor)
            
            return None

        return dfs(0, path, visited)

    # Try to find 3 cycles
    max_cycles = 3
    for _ in range(max_cycles):
        cycle = find_ham_cycle()
        if cycle is None:
            break
        
        # Add to result
        cycles.append(cycle)
        
        # Mark edges as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if u > v:
                u, v = v, u
            used_edges.add((u, v))
            
    # Ensure we return a non-empty list
    # The problem states "nonempty list". If we find 0, we must return at least 1.
    # But the graph is dense, we should find at least 1.
    # If for some reason we find 0, we return a dummy valid cycle? 
    # No, the rules say "Any broken rule makes the whole result invalid".
    # We must return valid cycles. If we can't find any, we fail.
    # But given the structure, we will find at least 1.
    
    return cycles

# EVOLVE-BLOCK-END
