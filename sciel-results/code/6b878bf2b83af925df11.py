import time
import numpy as np
from collections import deque

def construct():
    t0 = time.time()
    deadline = 85.0
    n = 16

    def _diameter_fast(adj, n):
        """Compute diameter using numpy matrix multiplication. Return None if disconnected."""
        # Use boolean matrix for speed
        A = np.array(adj, dtype=bool)
        # Check connectivity: A^2 + A should reach all nodes from each node
        # Actually, let's do BFS but with numpy for speed
        maxd = 0
        for s in range(n):
            dist = np.full(n, -1, dtype=np.int8)
            dist[s] = 0
            frontier = np.array([s], dtype=np.int32)
            d = 0
            while len(frontier) > 0:
                d += 1
                # Get neighbors of frontier
                neighbors = A[frontier]
                # Flatten and get unique
                if neighbors.size > 0:
                    new_frontier = np.where(neighbors.any(axis=1))[0]
                    # Filter out already visited
                    new_frontier = new_frontier[dist[new_frontier] == -1]
                    if len(new_frontier) == 0:
                        break
                    dist[new_frontier] = d
                    frontier = new_frontier
            if -1 in dist:
                return None
            maxd = max(maxd, int(np.max(dist)))
        return maxd

    def _is_edge_critical_fast(adj, n, u, v):
        """Check if edge (u,v) is critical using fast BFS."""
        adj[u][v] = 0
        adj[v][u] = 0
        
        # Check if graph is still connected and diameter <= 2
        A = np.array(adj, dtype=bool)
        maxd = 0
        for s in range(n):
            dist = np.full(n, -1, dtype=np.int8)
            dist[s] = 0
            frontier = np.array([s], dtype=np.int32)
            d = 0
            while len(frontier) > 0:
                d += 1
                if d > 2:
                    # Already exceeded diameter 2
                    adj[u][v] = 1
                    adj[v][u] = 1
                    return True
                neighbors = A[frontier]
                if neighbors.size > 0:
                    new_frontier = np.where(neighbors.any(axis=1))[0]
                    new_frontier = new_frontier[dist[new_frontier] == -1]
                    if len(new_frontier) == 0:
                        break
                    dist[new_frontier] = d
                    frontier = new_frontier
            if -1 in dist:
                # Disconnected
                adj[u][v] = 1
                adj[v][u] = 1
                return True
            maxd = max(maxd, int(np.max(dist)))
        
        adj[u][v] = 1
        adj[v][u] = 1
        return maxd > 2

    # Key insight: In a triangle-free graph, removing edge (u,v) means u and v have no common neighbor,
    # so their distance becomes >= 3 (or they become disconnected). This makes every edge critical.
    # K_{8,8} is triangle-free with diameter 2 and 64 edges.
    
    A_part = list(range(0, 8))
    B_part = list(range(8, 16))
    edges = [[a, b] for a in A_part for b in B_part]
    
    # Build adjacency
    adj = [[0]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    
    # Verify initial diameter is 2
    if _diameter_fast(adj, n) != 2:
        # Fallback: star graph (diameter 2, 15 edges)
        return [[0, v] for v in range(1, n)]
    
    # Verify every edge is critical
    for u, v in edges:
        if time.time() > deadline - 2:
            break
        if not _is_edge_critical_fast(adj, n, u, v):
            # This shouldn't happen for K_{8,8} since it's triangle-free
            # But if it does, fall back to star
            return [[0, v] for v in range(1, n)]
    
    # Return edges as list of [u, v] with Python int types
    result = []
    for u, v in edges:
        result.append([int(u), int(v)])
    
    return result
