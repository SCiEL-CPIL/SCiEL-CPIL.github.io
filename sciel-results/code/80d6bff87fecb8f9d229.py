import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    n = P['n']
    base_edges = set(tuple(e) for e in P['edges'])
    base_edges.update((v, u) for u, v in base_edges)
    
    candidates = []
    for u in range(n):
        for v in range(u + 2, n):
            if (u - v) % 2 != 0:
                if (u, v) not in base_edges:
                    candidates.append((u, v))
                    
    def get_diameter_and_worst(edges):
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_dist = 0
        worst_pair = (0, 1)
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                curr = queue.pop(0)
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        dist[nei] = dist[curr] + 1
                        queue.append(nei)
            for end in range(n):
                if dist[end] > max_dist:
                    max_dist = dist[end]
                    worst_pair = (start, end)
        return max_dist, worst_pair

    current_edges = list(base_edges)
    
    while time.time() < deadline:
        diam, (a, b) = get_diameter_and_worst(current_edges)
        if diam <= 4:
            break
            
        # Compute all-pairs distances for current_edges
        adj = {i: [] for i in range(n)}
        for u, v in current_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        dist = [[-1]*n for _ in range(n)]
        for start in range(n):
            dist[start][start] = 0
            queue = [start]
            while queue:
                curr = queue.pop(0)
                for nei in adj[curr]:
                    if dist[start][nei] == -1:
                        dist[start][nei] = dist[start][curr] + 1
                        queue.append(nei)
                        
        best_edge = None
        min_dist_ab = n
        # Tie-breaker: prefer larger span? Or just first found?
        for u, v in candidates:
            if (u, v) in current_edges or (v, u) in current_edges:
                continue
                
            d1 = dist[a][u] + 1 + dist[v][b]
            d2 = dist[a][v] + 1 + dist[u][b]
            new_d_ab = min(dist[a][b], d1, d2)
            
            if new_d_ab < min_dist_ab:
                min_dist_ab = new_d_ab
                best_edge = (u, v)
                
        if best_edge is None:
            break
            
        current_edges.append(best_edge)
        
    added_edges = [list(e) for e in current_edges if e not in base_edges]
    
    # Remove redundant edges
    final_edges = list(added_edges)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(final_edges)):
            temp_edges = [e for j, e in enumerate(final_edges) if j != i]
            all_temp = list(set(list(base_edges) + temp_edges))
            diam, _ = get_diameter_and_worst(all_temp)
            if diam <= 4:
                final_edges = temp_edges
                changed = True
                break
                
    return final_edges
