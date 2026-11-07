import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # 1. Check for large Max Clique (trivial subdivision)
    # Bron-Kerbosch
    max_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal max_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        # Pivot
        if P_set or X:
            pivot = max(list(P_set) + list(X), key=lambda v: len(adj_set[v] & P_set))
            candidates = [v for v in P_set if v not in adj_set[pivot]]
        else:
            candidates = list(P_set)
            
        for v in candidates:
            bron_kerbosch(R + [v], P_set & adj_set[v], X & adj_set[v])
            P_set.remove(v)
            X.add(v)
            
    bron_kerbosch([], set(range(n)), set())
    
    if len(max_clique) >= 7:
        branches = [int(v) for v in max_clique]
        paths = [[int(u), int(v)] for u, v in combinations(branches, 2)]
        return {'branches': branches, 'paths': paths}
    
    # 2. Search for clique subdivision of size k
    # We try k from 10 down to 3
    # Use tiered candidate pools to balance search space and quality
    
    pools = {
        10: vertices_by_degree[:14],
        9: vertices_by_degree[:16],
        8: vertices_by_degree[:18],
        7: vertices_by_degree[:20],
        6: vertices_by_degree[:22],
        5: vertices_by_degree[:24]
    }
    
    # Helper to find disjoint paths for a given set of branches
    def find_subdivision(branches, max_retries=10):
        branches_set = set(branches)
        pairs = list(combinations(branches, 2))
        num_pairs = len(pairs)
        
        # Pre-compute pair indices for quick lookup
        pair_to_idx = {}
        for i, (u, v) in enumerate(pairs):
            pair_to_idx[(u, v)] = i
            pair_to_idx[(v, u)] = i
            
        for attempt in range(max_retries):
            if time.time() > deadline:
                return None
            
            # Shuffle pairs
            shuffled_pairs = pairs[:]
            random.shuffle(shuffled_pairs)
            
            used_vertices = set()
            paths = [None] * num_pairs
            
            def find_paths(idx):
                if time.time() > deadline:
                    return False
                
                if idx == len(shuffled_pairs):
                    return True
                
                u, v = shuffled_pairs[idx]
                
                # BFS to find a path from u to v
                # We want a simple path.
                # To allow for different paths, we can randomize the order of neighbors in the BFS queue expansion
                # or just rely on the pair ordering shuffling.
                
                parent = [-1] * n
                visited = [False] * n
                queue = deque()
                queue.append(u)
                visited[u] = True
                
                found = False
                while queue and not found:
                    curr = queue.popleft()
                    if curr == v:
                        found = True
                        break
                    # Shuffle neighbors to get different shortest paths
                    neighbors = adj[curr][:]
                    random.shuffle(neighbors)
                    for nei in neighbors:
                        if not visited[nei]:
                            if nei in branches_set and nei != v:
                                continue
                            if nei in used_vertices:
                                continue
                            visited[nei] = True
                            parent[nei] = curr
                            queue.append(nei)
                
                if not found:
                    return False
                
                path = []
                curr = v
                while curr != -1:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                
                interior = set(path[1:-1])
                
                # Check disjointness
                if interior & used_vertices:
                    return False # Should not happen if BFS checked, but safe
                
                # Map back to original pair index
                orig_idx = pair_to_idx[(u, v)]
                paths[orig_idx] = path
                used_vertices.update(interior)
                
                if find_paths(idx + 1):
                    return True
                
                # Backtrack
                used_vertices.difference_update(interior)
                paths[orig_idx] = None
                return False
            
            if find_paths(0):
                # Success
                # Map paths back to original pair order
                result_paths = [list(map(int, p)) for p in paths if p is not None]
                return result_paths
            
        return None
    
    for k in range(10, 2, -1):
        if time.time() > deadline:
            break
            
        if k not in pools:
            continue
            
        candidates = pools[k]
        
        # Adjust retries based on k
        # Larger k means fewer combinations, so we can afford more retries per combo
        # Smaller k means more combinations, so fewer retries per combo
        retries = 50 if k >= 9 else 20
        
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
                
            branches = list(combo)
            paths = find_subdivision(branches, max_retries=retries)
            
            if paths:
                return {
                    'branches': [int(b) for b in branches],
                    'paths': paths
                }
    
    # Fallback: Max Clique found earlier
    if max_clique:
        branches = [int(v) for v in max_clique]
        paths = [[int(u), int(v)] for u, v in combinations(branches, 2)]
        return {'branches': branches, 'paths': paths}
    
    # Fallback: Any triangle
    es = {tuple(e) for e in edges}
    for row in combinations(range(n), 3):
        if all(e in es for e in combinations(row, 2)):
            return {
                'branches': [int(u) for u in row],
                'paths': [list(map(int, e)) for e in combinations(row, 2)]
            }
    
    # Absolute fallback
    u, v = edges[0]
    return {
        'branches': [int(u), int(v)],
        'paths': [[int(u), int(v)]]
    }
