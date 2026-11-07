import sys
from itertools import combinations
import time

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    def find_disjoint_paths(new_branch, existing_branches, used_interiors, adj, time_limit):
        forbidden = set(used_interiors) | set(existing_branches)
        
        candidate_paths = {}
        for b in existing_branches:
            queue = [(new_branch, [new_branch])]
            visited = {new_branch}
            found_paths = []
            while queue and len(found_paths) < 5:
                node, path = queue.pop(0)
                if node == b:
                    found_paths.append(path)
                    continue
                for neighbor in adj[node]:
                    if neighbor in visited:
                        continue
                    if neighbor in forbidden and neighbor != b:
                        continue
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
            
            if not found_paths:
                return None
            candidate_paths[b] = found_paths
        
        def backtrack(idx, current_paths, current_used):
            if time.time() > time_limit:
                return None
            if idx == len(existing_branches):
                return current_paths
            
            b = existing_branches[idx]
            for path in candidate_paths[b]:
                interior = set(path[1:-1])
                if interior & current_used:
                    continue
                
                new_used = current_used | interior
                res = backtrack(idx + 1, current_paths + [path], new_used)
                if res is not None:
                    return res
            return None
        
        return backtrack(0, [], set())

    vertices = list(range(n))
    vertices.sort(key=lambda x: len(adj[x]), reverse=True)
    
    # Try k from 10 down to 2
    for k in range(10, 1, -1):
        if time.time() > deadline:
            break
            
        if k > len(vertices):
            continue
            
        # Use top 15 vertices as candidates
        pool_size = min(15, n)
        candidate_pool = vertices[:pool_size]
        
        if k > len(candidate_pool):
            continue
            
        for branch_combo in combinations(candidate_pool, k):
            if time.time() > deadline:
                break
                
            branches = list(branch_combo)
            
            # Check validity
            b_sorted = sorted(branches, key=lambda x: len(adj[x]), reverse=True)
            used_interiors = set()
            paths_found = {}
            valid = True
            
            for i in range(len(b_sorted)):
                if time.time() > deadline:
                    valid = False
                    break
                b_new = b_sorted[i]
                existing = b_sorted[:i]
                
                if not existing:
                    continue
                
                res = find_disjoint_paths(b_new, existing, used_interiors, adj, deadline)
                if res is None:
                    valid = False
                    break
                
                for j, b_old in enumerate(existing):
                    path = res[j]
                    interior = set(path[1:-1])
                    if interior & used_interiors:
                        valid = False
                        break
                    used_interiors |= interior
                    paths_found[(b_new, b_old)] = path
                    paths_found[(b_old, b_new)] = path
                    
                if not valid:
                    break
            
            if valid:
                result_paths = []
                for b1, b2 in combinations(branches, 2):
                    if (b1, b2) in paths_found:
                        p = paths_found[(b1, b2)]
                    else:
                        p = paths_found[(b2, b1)]
                    result_paths.append(list(p))
                
                return {
                    'branches': list(branches),
                    'paths': result_paths
                }
                
    # Fallback: Triangle
    for row in combinations(range(n), 3):
        if all((u, v) in adj[u] for u, v in combinations(row, 2)):
            return {
                'branches': list(row),
                'paths': [list(e) for e in combinations(row, 2)]
            }
            
    u, v = edges[0]
    return {
        'branches': [u, v],
        'paths': [[u, v]]
    }
