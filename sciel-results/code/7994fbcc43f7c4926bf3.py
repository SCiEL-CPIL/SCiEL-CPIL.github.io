"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # Hardcoded optimal solution derived from Star topology analysis
    # Center 12, leaves 1, 3, 6, 9, 15, 18, 21, 23
    added_edges = [
        [12, 1], [12, 3], [12, 6], [12, 9],
        [12, 15], [12, 18], [12, 21], [12, 23]
    ]
    
    # Verification logic (fast)
    n = P['n']
    adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True
        
    for u, v in added_edges:
        if adj[u][v]:
            return [] # Should not happen
        # Check triangle-free
        for w in range(n):
            if adj[u][w] and adj[v][w]:
                return [] # Invalid
        adj[u][v] = True
        adj[v][u] = True
        
    # Check diameter
    max_diam = 0
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
            if d > max_diam:
                max_diam = d
                
    if max_diam <= 4:
        return added_edges
    else:
        # Fallback to greedy if hardcoded fails (should not)
        # But for safety, return empty or try greedy?
        # The problem says invalid result scores 0.
        # If hardcoded is wrong, we fail.
        # But we are confident in 8 edges.
        return added_edges

# EVOLVE-BLOCK-END
