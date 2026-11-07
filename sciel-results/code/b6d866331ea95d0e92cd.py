import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Helper to find candidate paths
    def find_candidate_paths(u, v, forbidden, branches_set):
        # BFS to find shortest paths
        # Returns list of paths (lists of vertices)
        if u == v:
            return [[u]]
            
        q = deque([(u, [u])])
        visited = {u}
        candidates = []
        
        while q and len(candidates) < 20:
            node, path = q.popleft()
            if node == v:
                candidates.append(path)
                if len(candidates) >= 10:
                    break
                continue
            for n in adj[node]:
                if n in visited or n in forbidden:
                    continue
                if n in branches_set and n != v:
                    continue
                visited.add(n)
                q.append((n, path + [n]))
                
        return candidates

    def check_subdivision(branches):
        branches_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Sort pairs by "difficulty": shorter paths first? Or fewer options?
        # Let's just use fixed order for now, or sort by distance in original graph.
        # Precomputing distances is expensive if done for every combo.
        # Let's just iterate.
        
        def backtrack(idx, used_internal, paths_so_far):
            if time.time() > deadline:
                return None
            
            if idx == len(pairs):
                return paths_so_far
            
            u, v = pairs[idx]
            
            # Forbidden: used internal vertices + other branches
            forbidden = set(used_internal)
            forbidden.update(branches_set)
            forbidden.discard(u)
            forbidden.discard(v)
            
            candidates = find_candidate_paths(u, v, forbidden, branches_set)
            
            if not candidates:
                return None
            
            # Try each candidate
            # Heuristic: try shortest first (BFS already does this)
            for p in candidates:
                new_used = set(used_internal)
                new_used.update(p[1:-1])
                
                res = backtrack(idx + 1, new_used, paths_so_far + [p])
                if res:
                    return res
                    
            return None
        
        return backtrack(0, set(), [])

    # Try to find max k
    # Upper bound: n
    # But K_k subdivision requires high connectivity.
    # Let's try from 8 down to 3.
    
    for k in range(8, 2, -1):
        if time.time() > deadline - 5:
            break
            
        found_paths = None
        found_branches = None
        
        # To speed up, we can filter combinations.
        # But let's just iterate.
        for comb in combinations(range(n), k):
            if time.time() > deadline - 5:
                break
                
            paths = check_subdivision(comb)
            if paths:
                found_paths = paths
                found_branches = list(comb)
                break
        
        if found_paths:
            # Construct result
            result = {
                'branches': found_branches,
                'paths': [list(p) for p in found_paths]
            }
            return result

    # Fallback: Find a triangle (k=3)
    es = {tuple(e) for e in P['edges']}
    for row in combinations(range(n), 3):
        if all(e in es for e in combinations(row, 2)):
            return {'branches': list(row), 'paths': [list(e) for e in combinations(row, 2)]}
            
    # Fallback: Single edge (k=2) - though task requires at least 2 branches, 
    # and "unordered pair" implies 1 path.
    u, v = P['edges'][0]
    return {'branches': [u, v], 'paths': [[u, v]]}
