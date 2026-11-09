import time
from collections import deque
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    base_edges = [(e[0], e[1]) for e in P['edges']]
    base_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in base_edges)
    
    def _build_adj(n, edge_list):
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def _is_triangle_free(n, edge_list):
        adj = _build_adj(n, edge_list)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True
    
    def _diameter(n, edge_list):
        adj = _build_adj(n, edge_list)
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            if -1 in dist:
                return None
            max_dist = max(max_dist, max(dist))
        return max_dist
    
    def _validate(n, base_edges, added_edges):
        all_edges = list(base_edges) + list(added_edges)
        seen = set()
        for u, v in all_edges:
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        if not _is_triangle_free(n, all_edges):
            return False
        d = _diameter(n, all_edges)
        if d is None or d > 4:
            return False
        return True
    
    # Generate all possible non-base edges
    all_possible = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set:
                all_possible.append((u, v))
    
    # Filter: exclude edges with distance 2 (immediate triangle with path)
    candidates = [(u, v) for u, v in all_possible if abs(u - v) >= 3]
    
    # Sort by length descending
    candidates.sort(key=lambda e: e[1] - e[0], reverse=True)
    
    best_added = None
    best_count = float('inf')
    
    random.seed(42)
    
    # Try k from 3 to 15
    for k in range(3, 16):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 100000 if k <= 5 else 50000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            if len(candidates) < k:
                continue
            
            # Select k random edges from candidates
            selected = random.sample(candidates, k)
            
            # Quick triangle-free check
            adj = _build_adj(n, base_edges + selected)
            tf = True
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        if adj[u] & adj[v]:
                            tf = False
                            break
                if not tf:
                    break
            
            if not tf:
                continue
            
            # Check diameter
            d = _diameter(n, base_edges + selected)
            if d is not None and d <= 4:
                if len(selected) < best_count:
                    best_added = [list(e) for e in selected]
                    best_count = len(selected)
                found = True
                break
        
        if found:
            break
    
    if best_added is not None:
        return [[int(u), int(v)] for u, v in best_added]
    
    return []

# EVOLVE-BLOCK-END
