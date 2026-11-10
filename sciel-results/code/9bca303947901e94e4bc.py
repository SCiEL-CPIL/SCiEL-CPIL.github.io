# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)
adj_list = [sorted(adj[i]) for i in range(N)]
deg = [len(adj[i]) for i in range(N)]

DEADLINE = time.time() + 50

def time_left():
    return DEADLINE - time.time()

def find_paths_backtrack(branches, pairs, used_interior_mask, path_indices):
    if not pairs:
        return path_indices
    
    # Heuristic: pick pair with fewest options
    best_pair_idx = 0
    min_options = float('inf')
    branch_set = set(branches)
    
    for idx, (i, j) in enumerate(pairs):
        u = branches[i]
        v = branches[j]
        rem_u = 0
        rem_v = 0
        for w in adj_list[u]:
            if w == v: continue
            if w in branch_set and w != u and w != v: continue
            if (used_interior_mask >> w) & 1: continue
            rem_u += 1
        for w in adj_list[v]:
            if w == u: continue
            if w in branch_set and w != u and w != v: continue
            if (used_interior_mask >> w) & 1: continue
            rem_v += 1
        options = rem_u + rem_v
        if options < min_options:
            min_options = options
            best_pair_idx = idx
            
    i, j = pairs[best_pair_idx]
    u, v = branches[i], branches[j]
    remaining_pairs = [p for idx, p in enumerate(pairs) if idx != best_pair_idx]
    
    paths = []
    
    def dfs(curr, target, path, visited_mask):
        if curr == target:
            paths.append(list(path))
            return
        for w in adj_list[curr]:
            if (visited_mask >> w) & 1:
                continue
            if w in branch_set and w != target and w != u:
                continue
            if (used_interior_mask >> w) & 1:
                continue
            visited_mask_new = visited_mask | (1 << w)
            path.append(w)
            dfs(w, target, path, visited_mask_new)
            path.pop()
            
    dfs(u, v, [u], 1 << u)
    paths.sort(key=lambda x: len(x))
    
    for p in paths:
        if len(p) < 2: continue
        new_used_mask = used_interior_mask
        valid = True
        for node in p[1:-1]:
            if (new_used_mask >> node) & 1:
                valid = False
                break
            new_used_mask |= (1 << node)
        if not valid:
            continue
            
        result = find_paths_backtrack(branches, remaining_pairs, new_used_mask, path_indices + [p])
        if result is not None:
            return result
            
    return None

def is_feasible(branches):
    if len(branches) < 2:
        return []
    pairs = list(combinations(range(len(branches)), 2))
    res = find_paths_backtrack(branches, pairs, 0, [])
    return res

def construct():
    global DEADLINE
    DEADLINE = time.time() + 50
    
    order = sorted(range(N), key=lambda x: -deg[x])
    
    best_branches = []
    best_paths = None
    current_branches = []
    
    for v in order:
        if time_left() < 2:
            break
        trial = current_branches + [v]
        paths = is_feasible(trial)
        if paths is not None:
            current_branches = trial
            if len(current_branches) > len(best_branches):
                best_branches = list(current_branches)
                best_paths = list(paths)
                
    # Local Search
    improved = True
    while improved and time_left() > 3:
        improved = False
        b_set = set(best_branches)
        for v in range(N):
            if v in b_set:
                continue
            trial = best_branches + [v]
            paths = is_feasible(trial)
            if paths is not None:
                best_branches = trial
                best_paths = paths
                improved = True
                break
                
    if not best_branches:
        return {'branches': [0, 1], 'paths': [[0, 1]]}
        
    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(v) for v in p] for p in best_paths]
    
    return {'branches': branches_out, 'paths': paths_out}
# EVOLVE-BLOCK-END
