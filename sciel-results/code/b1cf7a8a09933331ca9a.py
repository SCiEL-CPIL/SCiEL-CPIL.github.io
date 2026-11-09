import time
from collections import deque
from itertools import combinations
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]
    degrees = [len(adj[i]) for i in range(n)]

    def time_ok():
        return time.time() < deadline

    def find_paths(s, t, forbidden_mask, max_len=7):
        if s == t:
            return [[s]]
        
        paths = []
        
        def dfs(node, path, visited_mask):
            if not time_ok():
                return
            if node == t:
                paths.append(list(path))
                return
            if len(path) - 1 >= max_len:
                return
            
            for nb in adj_list[node]:
                if nb == t:
                    new_path = path + [t]
                    paths.append(new_path)
                    continue
                
                if nb in visited_mask:
                    continue
                if nb in forbidden_mask:
                    continue
                
                visited_mask |= (1 << nb)
                path.append(nb)
                dfs(nb, path, visited_mask)
                path.pop()
                visited_mask ^= (1 << nb)
                
        visited = (1 << s)
        dfs(s, [s], visited)
        
        paths.sort(key=len)
        return paths

    def solve_disjoint_paths(branches):
        k = len(branches)
        if k < 2:
            return None
        
        branches_set = set(branches)
        pairs = []
        for i in range(k):
            for j in range(i+1, k):
                pairs.append((branches[i], branches[j]))
        
        pair_info = []
        for u, v in pairs:
            forb_mask = 0
            for b in branches_set:
                if b != u and b != v:
                    forb_mask |= (1 << b)
            pair_info.append((u, v, forb_mask))
        
        def backtrack(assigned, used_mask, remaining_indices):
            if not time_ok():
                return None
            
            if not remaining_indices:
                # Convert assigned indices to (u,v) keys
                result = {}
                for idx, path in assigned.items():
                    u, v, _ = pair_info[idx]
                    result[(u, v)] = path
                return result
            
            best_idx = -1
            best_paths = None
            min_count = float('inf')
            
            for idx in remaining_indices:
                u, v, forb_mask = pair_info[idx]
                current_forb = forb_mask | used_mask
                
                candidates = find_paths(u, v, current_forb, max_len=6)
                
                if len(candidates) < min_count:
                    min_count = len(candidates)
                    best_idx = idx
                    best_paths = candidates
                
                if min_count == 0:
                    return None
            
            if best_idx == -1:
                return None

            u, v, _ = pair_info[best_idx]
            new_remaining = [i for i in remaining_indices if i != best_idx]
            
            for path in best_paths:
                interior_mask = 0
                for node in path[1:-1]:
                    interior_mask |= (1 << node)
                
                new_used_mask = used_mask | interior_mask
                new_assigned = dict(assigned)
                new_assigned[best_idx] = path
                
                result = backtrack(new_assigned, new_used_mask, new_remaining)
                if result is not None:
                    return result
            
            return None

        return backtrack({}, 0, list(range(len(pairs))))

    def find_max_clique():
        best = []
        
        def bk(R, Pset, X):
            nonlocal best
            if not time_ok():
                return
            if not Pset and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            
            if Pset:
                pivot_candidates = list(Pset | X)
            else:
                pivot_candidates = list(X)
            
            if not pivot_candidates:
                return
            
            u = max(pivot_candidates, key=lambda v: len(adj[v] & Pset))
            candidates = [v for v in Pset if v not in adj[u]]
            candidates.sort(key=lambda x: -degrees[x])
            
            for v in candidates:
                if not time_ok():
                    return
                bk(R | {v}, Pset & adj[v], X & adj[v])
                Pset = Pset - {v}
                X = X | {v}
        
        bk(frozenset(), frozenset(range(n)), frozenset())
        return best

    max_clique = find_max_clique()
    best_result = None
    best_k = 0

    if len(max_clique) >= 2:
        clique_list = sorted(max_clique)
        paths_dict = solve_disjoint_paths(clique_list)
        if paths_dict:
            best_result = (clique_list, paths_dict)
            best_k = len(clique_list)

    if best_result is None or best_k < len(max_clique):
        for size in range(min(len(max_clique), 6), 1, -1):
            if not time_ok():
                break
            found = False
            for _ in range(50):
                if not time_ok():
                    break
                subset = list(random.sample(max_clique, size))
                subset.sort()
                paths_dict = solve_disjoint_paths(subset)
                if paths_dict:
                    best_result = (subset, paths_dict)
                    best_k = size
                    found = True
                    break
            if found:
                break
        
        if best_result is None:
            for u, v in edges:
                if not time_ok():
                    break
                paths_dict = solve_disjoint_paths([u, v])
                if paths_dict:
                    best_result = ([u, v], paths_dict)
                    best_k = 2
                    break
            
            if best_result is None:
                return {'branches': [0, 1], 'paths': [[0, 1]]}

    current_branches = list(best_result[0])
    current_paths = best_result[1]
    
    improved = True
    while improved and time_ok():
        improved = False
        candidates = [v for v in range(n) if v not in current_branches]
        random.shuffle(candidates)
        
        for v in candidates:
            if not time_ok():
                break
            
            new_branches = sorted(current_branches + [v])
            new_paths = solve_disjoint_paths(new_branches)
            
            if new_paths:
                current_branches = new_branches
                current_paths = new_paths
                best_result = (current_branches, current_paths)
                best_k = len(current_branches)
                improved = True
                break

    branches_out = [int(b) for b in current_branches]
    paths_out = []
    k = len(branches_out)
    
    for i in range(k):
        for j in range(i+1, k):
            u, v = branches_out[i], branches_out[j]
            p = current_paths.get((u, v))
            if p is None:
                p = current_paths.get((v, u))
                if p:
                    p = p[::-1]
            if p:
                paths_out.append([int(x) for x in p])

    return {
        'branches': branches_out,
        'paths': paths_out
    }

# Run construct to test
# print(construct())
