"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    
    # Initialize adjacency and edge set
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u, v), max(u, v)))
        
    added_edges = []
    
    def get_dist_matrix():
        dist = [[-1] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            q = deque([i])
            while q:
                node = q.popleft()
                for nei in adj[node]:
                    if dist[i][nei] == -1:
                        dist[i][nei] = dist[i][node] + 1
                        q.append(nei)
        return dist

    def get_diameter(dist):
        max_d = 0
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
        return max_d

    def is_connected():
        visited = [False] * n
        q = deque([0])
        visited[0] = True
        count = 1
        while q:
            node = q.popleft()
            for nei in adj[node]:
                if not visited[nei]:
                    visited[nei] = True
                    count += 1
                    q.append(nei)
        return count == n

    def would_create_triangle(u, v):
        for w in adj[u]:
            if w in adj[v]:
                return True
        return False

    def try_add_edge(u, v):
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u, v), max(u, v)))
        added_edges.append([u, v])

    def try_remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        edge_set.remove((min(u, v), max(u, v)))
        added_edges.remove([u, v])

    # Phase 1: Greedy Add
    while True:
        dist = get_dist_matrix()
        d = get_diameter(dist)
        if d <= 4:
            break
        
        best_edge = None
        best_diam = d + 1
        
        for u in range(n):
            for v in range(u+1, n):
                if (min(u, v), max(u, v)) in edge_set:
                    continue
                if would_create_triangle(u, v):
                    continue
                
                new_max_d = 0
                for i in range(n):
                    for j in range(i+1, n):
                        old_d = dist[i][j]
                        d1 = dist[i][u] + 1 + dist[v][j]
                        d2 = dist[i][v] + 1 + dist[u][j]
                        new_d = old_d
                        if d1 < new_d:
                            new_d = d1
                        if d2 < new_d:
                            new_d = d2
                        
                        if new_d > new_max_d:
                            new_max_d = new_d
                            if new_max_d >= best_diam:
                                break
                    if new_max_d >= best_diam:
                        break
                
                if new_max_d < best_diam:
                    best_diam = new_max_d
                    best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        try_add_edge(u, v)

    # Phase 2: Greedy Remove
    changed = True
    while changed:
        changed = False
        for edge in list(added_edges):
            u, v = edge
            try_remove_edge(u, v)
            
            if not is_connected():
                try_add_edge(u, v)
                continue
            
            dist = get_dist_matrix()
            d = get_diameter(dist)
            
            if d > 4:
                try_add_edge(u, v)
            else:
                changed = True

    return added_edges
# EVOLVE-BLOCK-END
