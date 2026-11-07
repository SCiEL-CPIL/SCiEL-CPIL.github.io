"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # Hardcoded solution with 8 edges
    added_edges = [
        [0, 11],
        [1, 12],
        [22, 11],
        [23, 12],
        [6, 17],
        [17, 22],
        [6, 1],
        [12, 17]
    ]
    
    # Verification logic (optional but good for safety)
    # Check types
    if not isinstance(added_edges, list):
        return []
    for e in added_edges:
        if not isinstance(e, list) or len(e) != 2:
            return []
        if not all(isinstance(x, int) for x in e):
            return []
        
    base = {tuple(e) for e in P['edges']}
    base |= {(v, u) for u, v in base}
    
    # Check duplicates and existing
    for u, v in added_edges:
        if (u, v) in base or (v, u) in base:
            return []
        if u == v:
            return []
            
    # Check triangle-free (bipartite)
    # Path is bipartite. Added edges must be between partitions.
    # Partition A: even, B: odd
    for u, v in added_edges:
        if (u % 2) == (v % 2):
            return []
            
    # Check diameter
    # Build adjacency
    adj = {i: set() for i in range(P['n'])}
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # BFS from each node to find max distance
    max_dist = 0
    for start in range(P['n']):
        visited = {start}
        queue = [(start, 0)]
        while queue:
            node, dist = queue.pop(0)
            max_dist = max(max_dist, dist)
            for nei in adj[node]:
                if nei not in visited:
                    visited.add(nei)
                    queue.append((nei, dist + 1))
                    
    if max_dist > 4:
        return []
        
    return added_edges
# EVOLVE-BLOCK-END
