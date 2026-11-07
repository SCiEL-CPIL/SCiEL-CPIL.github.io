"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    
    # State
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u, v), max(u, v)))
        
    added_edges = []

    def get_diameter():
        max_d = 0
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                node = q.popleft()
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        q.append(nei)
                        if dist[nei] > max_d:
                            max_d = dist[nei]
        return max_d

    def get_all_dists():
        dists = []
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                node = q.popleft()
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        q.append(nei)
            dists.append(dist)
        return dists

    # Generate candidate edges: only between even and odd vertices to ensure triangle-free
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in edge_set:
                continue
            if u % 2 == v % 2:
                continue
            candidates.append((u, v))

    # Phase 1: Greedy Add
    while True:
        d = get_diameter()
        if d <= 4:
            break
            
        dists = get_all_dists()
        
        best_score = -1
        best_edge = None
        
        for u, v in candidates:
            if (u, v) in edge_set:
                continue
                
            # Score: number of pairs (i, j) with dist > 4 that are reduced to <= 4
            # New dist(i, j) = min(old, dist(i, u) + 1 + dist(v, j), dist(i, v) + 1 + dist(u, j))
            # We only care if old > 4 and new <= 4
            
            score = 0
            for i in range(n):
                for j in range(i + 1, n):
                    old_d = dists[i][j]
                    if old_d <= 4:
                        continue
                    new_d = min(old_d, 
                                dists[i][u] + 1 + dists[v][j],
                                dists[i][v] + 1 + dists[u][j])
                    if new_d <= 4:
                        score += 1
                        
            if score > best_score:
                best_score = score
                best_edge = (u, v)
                
        if best_edge is None or best_score == 0:
            # If no edge helps, we might be stuck, but for this problem it shouldn't happen
            break
            
        u, v = best_edge
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v))
        added_edges.append([u, v])

    # Phase 2: Greedy Remove
    changed = True
    while changed:
        changed = False
        for edge in list(added_edges):
            u, v = edge
            # Temporarily remove
            adj[u].remove(v)
            adj[v].remove(u)
            edge_set.remove((u, v))
            added_edges.remove([u, v])
            
            d = get_diameter()
            if d > 4:
                # Restore
                adj[u].append(v)
                adj[v].append(u)
                edge_set.add((u, v))
                added_edges.append([u, v])
            else:
                changed = True

    return added_edges
# EVOLVE-BLOCK-END
