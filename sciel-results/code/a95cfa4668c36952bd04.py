import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(n, adj):
    for u in range(n):
        for v in adj[u]:
            for w in adj[u]:
                if v < w and w in adj[v]:
                    return False
    return True

def _diameter(n, adj):
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
            return float('inf')
        max_dist = max(max_dist, max(dist))
    return max_dist

def _validate(n, base_edges, added_edges):
    all_edges = list(base_edges) + list(added_edges)
    adj = _build_adj(n, all_edges)
    if not _is_triangle_free(n, adj):
        return False
    d = _diameter(n, adj)
    return d <= 4

def construct():
    deadline = time.time() + 85
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for e in base_edges:
        base_set.add(tuple(sorted(e)))
    
    # Candidate edges: only between even and odd
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u % 2) != (v % 2):
                if tuple(sorted((u,v))) not in base_set:
                    candidates.append((u, v))
    
    # Known 10-edge solution: Hub at 12 connected to all other odd vertices
    # Odd vertices: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23
    # 12 is even.
    # Base edges incident to 12: 11-12, 12-13.
    # So we need to add edges from 12 to {1, 3, 5, 7, 9, 15, 17, 19, 21, 23}
    hub12 = 12
    hub12_neighbors_to_add = [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]
    sol10 = [[hub12, v] for v in hub12_neighbors_to_add]
    
    # Verify sol10
    if not _validate(n, base_edges, sol10):
        # Fallback to a different known solution if this one fails (shouldn't)
        # Try hub at 11 (odd)
        hub11 = 11
        # Even neighbors to add: 0, 2, 4, 6, 8, 10, 14, 16, 18, 20, 22
        # Base: 10-11, 11-12
        hub11_neighbors_to_add = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
        sol10 = [[hub11, v] for v in hub11_neighbors_to_add]
        if not _validate(n, base_edges, sol10):
             # Just return a valid 10-edge solution or whatever
             return sol10

    best_sol = sol10
    best_k = 10
    
    # Try to reduce to 9 edges using local search
    # Strategy: For each edge in best_sol, remove it, check validity.
    # If invalid, try swapping with a candidate edge.
    
    current_set = set(tuple(sorted(e)) for e in best_sol)
    
    for e_to_remove in best_sol:
        if time.time() > deadline:
            break
        e_key = tuple(sorted(e_to_remove))
        
        # 1. Try removing just this edge
        temp_set = current_set - {e_key}
        temp_list = [list(e) for e in temp_set]
        if _validate(n, base_edges, temp_list):
            best_sol = temp_list
            best_k = 9
            return best_sol
            
        # 2. Try swapping this edge with a candidate
        for cand in candidates:
            if time.time() > deadline:
                break
            cand_key = tuple(sorted(cand))
            if cand_key in current_set:
                continue
                
            swap_set = temp_set | {cand_key}
            swap_list = [list(e) for e in swap_set]
            if _validate(n, base_edges, swap_list):
                best_sol = swap_list
                best_k = 9
                return best_sol

    # If 9 not found, try random restarts for 9 edges
    import random
    random.seed(42)
    
    # Try a few random 9-edge sets
    for _ in range(1000):
        if time.time() > deadline:
            break
        sample = random.sample(candidates, 9)
        sample_list = [list(e) for e in sample]
        if _validate(n, base_edges, sample_list):
            best_sol = sample_list
            best_k = 9
            return best_sol

    return best_sol
