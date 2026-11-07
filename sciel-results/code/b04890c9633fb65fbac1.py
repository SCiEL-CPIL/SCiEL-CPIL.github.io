"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    
    # Pre-defined solution: 10 edges connecting independent sets to center 12
    # Left side: 0, 2, 4, 6, 8
    # Right side: 15, 17, 19, 21, 23
    added_edges = [
        [0, 12], [2, 12], [4, 12], [6, 12], [8, 12],
        [15, 12], [17, 12], [19, 12], [21, 12], [23, 12]
    ]
    
    # Build adjacency matrix for verification
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u,v), max(u,v)))
        
    for u, v in added_edges:
        if (min(u,v), max(u,v)) in edge_set:
            return []
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u,v), max(u,v)))
        
    # Verify triangle-free
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                for w in range(v+1, n):
                    if adj[u][w] and adj[v][w]:
                        return []
                        
    # Verify diameter
    def bfs_diameter(adj):
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in range(n):
                    if adj[u][v] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d > max_d:
                    max_d = d
        return max_d
        
    if bfs_diameter(adj) <= 4:
        return added_edges
    else:
        return []

# EVOLVE-BLOCK-END
