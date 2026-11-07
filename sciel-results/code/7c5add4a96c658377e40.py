import time
import random
from collections import deque

DEADLINE = 90.0
START_TIME = time.time()

def time_left():
    return DEADLINE - (time.time() - START_TIME)

def bfs_distances(adj, n):
    """Compute all-pairs shortest paths using BFS from each vertex."""
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

def graph_diameter(dist, n):
    """Return diameter, or -1 if disconnected."""
    max_d = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dist[i][j] == -1:
                return -1  # disconnected
            max_d = max(max_d, dist[i][j])
    return max_d

def is_valid_graph(edges, n=16):
    """Check if graph has diameter exactly 2 and every edge is critical."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check diameter is exactly 2
    dist = bfs_distances(adj, n)
    if graph_diameter(dist, n) != 2:
        return False
    
    # Check each edge is critical
    for (u, v) in edges:
        # Remove edge (u, v)
        adj2 = [set(s) for s in adj]
        adj2[u].discard(v)
        adj2[v].discard(u)
        
        dist2 = bfs_distances(adj2, n)
        diam2 = graph_diameter(dist2, n)
        if diam2 == -1:
            continue  # disconnected, OK
        if diam2 > 2:
            continue  # diameter increased, OK
        # Otherwise, edge is not critical
        return False
    
    return True

def construct():
    n = 16
    best_edges = None
    best_count = 0
    
    # Start with star
    star_edges = [[0, v] for v in range(1, n)]
    if is_valid_graph(star_edges, n):
        best_edges = star_edges[:]
        best_count = len(star_edges)
    
    # Try to improve by adding edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    current = star_edges[:]
    
    # Greedy: try adding each possible edge if it maintains validity
    for u, v in all_possible:
        if (u, v) in [(e[0], e[1]) for e in current] or (v, u) in [(e[0], e[1]) for e in current]:
            continue
        if time_left() < 10:
            break
        test = current + [[u, v]]
        if is_valid_graph(test, n):
            current = test
            if len(current) > best_count:
                best_count = len(current)
                best_edges = current[:]
    
    # Local search: try swapping edges
    if best_edges is not None:
        current = best_edges[:]
        improved = True
        while improved and time_left() > 5:
            improved = False
            current_set = set((min(u,v), max(u,v)) for u,v in current)
            for u, v in all_possible:
                if (min(u,v), max(u,v)) in current_set:
                    continue
                if time_left() < 5:
                    break
                # Try adding
                test = current + [[u, v]]
                if is_valid_graph(test, n):
                    current = test
                    current_set = set((min(a,b), max(a,b)) for a,b in current)
                    if len(current) > best_count:
                        best_count = len(current)
                        best_edges = current[:]
                    improved = True
                    break
    
    # Convert to proper format
    result = []
    for u, v in best_edges:
        u = int(u)
        v = int(v)
        result.append([u, v])
    
    return result
