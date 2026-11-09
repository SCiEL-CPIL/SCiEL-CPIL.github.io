import time
import collections
import random
import itertools

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 58.0

    n = P['n']
    edges = P['edges']
    adj = collections.defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = {u: list(v) for u, v in adj.items()}
    for i in range(n):
        if i not in adj_list:
            adj_list[i] = []
    
    # Precompute all simple paths between every pair of vertices
    # Limit path length to avoid explosion (e.g., max 6 vertices in path)
    all_paths = {}
    for u in range(n):
        for v in range(u + 1, n):
            paths = []
            def dfs(curr, target, path, visited):
                if len(path) > 6:  # Limit path length
                    return
                if curr == target:
                    paths.append(path[:])
                    return
                for nei in adj_list[curr]:
                    if nei not in visited:
                        visited.add(nei)
                        path.append(nei)
                        dfs(nei, target, path, visited)
                        path.pop()
                        visited.remove(nei)
            dfs(u, v, [u], {u})
            all_paths[(u, v)] = paths
    
    def check_subdivision(branches):
        k = len(branches)
        if k < 2:
            return None
        
        branch_set = set(branches)
        pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
        
        assigned_paths = {}
        
        def backtrack(idx, used_internal):
            if time.time() > deadline - 0.5:
                return False
            if idx == len(pairs):
                return True
                
            i, j = pairs[idx]
            u = branches[i]
            v = branches[j]
            
            # Get precomputed paths for this pair
            if u > v:
                u, v = v, u
            candidate_paths = all_paths.get((u, v), [])
            
            if not candidate_paths:
                return False
            
            # Filter paths that don't use any used_internal vertices or other branch vertices
            valid_paths = []
            for p in candidate_paths:
                interior = p[1:-1]
                if any(node in used_internal for node in interior):
                    continue
                if any(node in branch_set for node in interior):
                    continue
                valid_paths.append(p)
            
            if not valid_paths:
                return False
            
            # Sort by length (prefer shorter to leave room for others)
            valid_paths.sort(key=len)
            
            for p in valid_paths:
                new_used = set(used_internal)
                for node in p[1:-1]:
                    new_used.add(node)
                
                assigned_paths[(i, j)] = p
                
                if backtrack(idx + 1, new_used):
                    return True
            
            return False

        if backtrack(0, set()):
            result_paths = []
            for i in range(k):
                for j in range(i+1, k):
                    result_paths.append(assigned_paths[(i, j)])
            return result_paths
        return None

    # Find maximum clique
    def find_max_clique():
        max_clique = []
        candidates = list(range(n))
        
        def expand(clique, candidates):
            nonlocal max_clique
            if len(clique) > len(max_clique):
                max_clique = clique[:]
            if not candidates:
                return
            for i in range(len(candidates)):
                v = candidates[i]
                new_candidates = [u for u in candidates[i+1:] if u in adj[v]]
                expand(clique + [v], new_candidates)
                
        expand([], candidates)
        return max_clique

    max_clique = find_max_clique()
    best_k = len(max_clique)
    best_branches = max_clique[:]
    best_paths = [[max_clique[i], max_clique[j]] for i in range(best_k) for j in range(i+1, best_k)]
    
    deg = [len(adj_list[i]) for i in range(n)]
    sorted_by_deg = sorted(range(n), key=lambda x: -deg[x])
    
    def try_set(cand):
        nonlocal best_k, best_branches, best_paths
        paths = check_subdivision(list(cand))
        if paths:
            k = len(cand)
            if k > best_k:
                best_k = k
                best_branches = list(cand)
                best_paths = paths
                return True
        return False

    # Phase 1: Greedy Expansion
    current_set = set(best_branches)
    for _ in range(100):
        if time.time() > deadline - 10.0:
            break
        
        candidates = []
        for v in range(n):
            if v in current_set:
                continue
            score = sum(1 for u in current_set if u in adj[v]) + deg[v]
            candidates.append((score, v))
        
        if not candidates:
            break
            
        candidates.sort(reverse=True)
        
        added = False
        for score, v in candidates[:5]:
            new_set = list(current_set) + [v]
            if try_set(new_set):
                current_set.add(v)
                added = True
                break
        
        if not added:
            break

    # Phase 2: Random sampling of larger sets
    target_k = best_k + 1
    pool_size = min(n, int(target_k * 2.5))
    pool = sorted_by_deg[:pool_size]
    
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 500:
        attempts += 1
        sample = random.sample(pool, target_k)
        sample.sort()
        
        if try_set(sample):
            target_k = best_k + 1
            attempts = 0
            
    # Phase 3: Random restarts from whole graph
    target_k = best_k + 1
    attempts = 0
    while time.time() < deadline - 2.0 and attempts < 200:
        attempts += 1
        sample = random.sample(range(n), target_k)
        sample.sort()
        if try_set(sample):
            target_k = best_k + 1
            attempts = 0

    # Ensure types
    if best_paths:
        return {
            'branches': [int(b) for b in best_branches],
            'paths': [[int(x) for x in p] for p in best_paths]
        }
    else:
        return {
            'branches': [int(b) for b in best_branches],
            'paths': [[int(best_branches[i]), int(best_branches[j])] for i in range(len(best_branches)) for j in range(i+1, len(best_branches))]
        }

# EVOLVE-BLOCK-END
