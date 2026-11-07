import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 50.0
    
    n = P['n']
    
    adj = [set() for _ in range(n)]
    existing_edges = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        existing_edges.add((min(u, v), max(u, v)))
    
    def bfs_all_pairs():
        dist = [[-1] * n for _ in range(n)]
        for src in range(n):
            dist[src][src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[src][v] == -1:
                        dist[src][v] = dist[src][u] + 1
                        queue.append(v)
        max_d = 0
        for i in range(n):
            for j in range(n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
        return dist, max_d
    
    added_edges = []
    
    while True:
        if time.time() > deadline:
            break
        
        dist, diam = bfs_all_pairs()
        if diam <= 4:
            break
        
        best_edge = None
        best_new_diam = diam + 1
        
        candidates = []
        for a in range(n):
            for b in range(a + 1, n):
                if (a % 2) == (b % 2):
                    continue
                key = (min(a, b), max(a, b))
                if key in existing_edges:
                    continue
                path_dist = abs(a - b)
                candidates.append((path_dist, a, b))
        
        candidates.sort(reverse=True)
        
        for path_dist, a, b in candidates:
            if time.time() > deadline:
                break
            
            adj[a].add(b)
            adj[b].add(a)
            existing_edges.add((min(a, b), max(a, b)))
            
            _, new_diam = bfs_all_pairs()
            
            adj[a].remove(b)
            adj[b].remove(a)
            existing_edges.remove((min(a, b), max(a, b)))
            
            if new_diam < best_new_diam:
                best_new_diam = new_diam
                best_edge = (a, b)
                if best_new_diam <= 4:
                    break
        
        if best_edge is None:
            break
        
        a, b = best_edge
        adj[a].add(b)
        adj[b].add(a)
        existing_edges.add((min(a, b), max(a, b)))
        added_edges.append([a, b])
    
    result = [[int(u), int(v)] for u, v in added_edges]
    return result
