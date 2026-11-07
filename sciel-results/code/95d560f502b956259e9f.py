import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin for 60s limit
    
    n = P['n']
    adj = {i: set() for i in range(n)}
    base_edges_set = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        base_edges_set.add((min(u, v), max(u, v)))
    
    added_edges = []
    
    def is_triangle_free(u, v):
        common = adj[u].intersection(adj[v])
        if common:
            return False
        return True
        
    def get_diameter():
        max_dist = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                curr = q.popleft()
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        dist[nei] = dist[curr] + 1
                        q.append(nei)
            if -1 in dist:
                return float('inf')
            max_dist = max(max_dist, max(dist))
        return max_dist
        
    def get_farthest_pair():
        max_dist = 0
        s, t = 0, 0
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                curr = q.popleft()
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        dist[nei] = dist[curr] + 1
                        q.append(nei)
            for j in range(n):
                if dist[j] > max_dist:
                    max_dist = dist[j]
                    s, t = i, j
        return s, t, max_dist

    def fallback_solution():
        # Return edges of K_{12,12} minus existing path edges
        # A = 0..11, B = 12..23
        edges = []
        for u in range(12):
            for v in range(12, 24):
                if (min(u, v), max(u, v)) not in base_edges_set:
                    edges.append([u, v])
        return edges

    # Greedy Search
    while True:
        if time.time() > deadline:
            break
            
        s, t, d = get_farthest_pair()
        if d <= 4:
            break
            
        # Find best edge to add
        best_edge = None
        best_score = float('inf')
        
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if v in adj[u]:
                    continue
                if not is_triangle_free(u, v):
                    continue
                candidates.append((u, v))
                
        if not candidates:
            break
            
        for u, v in candidates:
            adj[u].add(v)
            adj[v].add(u)
            
            new_d = get_diameter()
            
            adj[u].remove(v)
            adj[v].remove(u)
            
            # We want to minimize diameter.
            # Tie-breaker: prefer edges that are "longer" in the original path?
            # Or just pick the first one with min diameter.
            if new_d < best_score:
                best_score = new_d
                best_edge = (u, v)
                
        if best_edge is None:
            break
            
        # If the best edge doesn't reduce the diameter, we might be stuck.
        # If best_score >= d, adding it doesn't help the global diameter.
        # In this case, we should probably abort and use fallback, 
        # OR keep adding if we hope it helps later?
        # Given the risk, if d > 4 and best_score >= d, let's try one more step?
        # Actually, if d=5 and best_score=5, we are stuck.
        # Let's check if we are making progress.
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added_edges.append([u, v])
        
        # If we added an edge but diameter didn't drop, and we've added many edges, 
        # it's likely we are converging to a dense graph or stuck.
        # If len(added_edges) > 100, it's likely not optimal.
        # But let's rely on the time check and final validation.

    # Final Validation
    final_d = get_diameter()
    if final_d <= 4:
        return added_edges
    else:
        # Fallback to K_{12,12}
        return fallback_solution()
