import random
import time

P = {'n': 16}

def has_diameter_2(adj):
    n = len(adj)
    for i in range(n):
        dist = [-1] * n
        dist[i] = 0
        queue = [i]
        while queue:
            u = queue.pop(0)
            for v in range(n):
                if adj[u][v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > 2:
                        return False
                    queue.append(v)
        if any(d == -1 for d in dist):
            return False
    return True

def is_edge_critical(adj, u, v):
    n = len(adj)
    adj_copy = [row[:] for row in adj]
    adj_copy[u][v] = 0
    adj_copy[v][u] = 0
    return not has_diameter_2(adj_copy)

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Start with a random geometric graph
    random.seed(42)
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i][j] = 1
                adj[j][i] = 1
    
    # Ensure diameter 2
    while not has_diameter_2(adj) and time.time() < deadline:
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j] == 0:
                    adj[i][j] = 1
                    adj[j][i] = 1
                    if has_diameter_2(adj):
                        break
            if has_diameter_2(adj):
                break
    
    # Remove non-critical edges
    critical_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 1 and is_edge_critical(adj, i, j):
                critical_edges.append([i, j])
    
    # Verify the final graph
    final_adj = [[0] * n for _ in range(n)]
    for u, v in critical_edges:
        final_adj[u][v] = 1
        final_adj[v][u] = 1
    
    if has_diameter_2(final_adj):
        return critical_edges
    else:
        # Fallback: return a known valid graph
        return [[0, v] for v in range(1, n)]
