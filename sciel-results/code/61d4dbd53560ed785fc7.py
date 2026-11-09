import time
import collections
import random

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
    # Convert to list for iteration speed
    adj_list = {u: list(v) for u, v in adj.items()}
    for i in range(n):
        if i not in adj_list:
            adj_list[i] = []

    def find_max_clique():
        max_clique = []
        candidates = list(range(n))
        
        def expand(clique, candidates):
            nonlocal max_clique
            if len(clique) > len(max_clique):
                max_clique = clique[:]
            if not candidates:
                return
            # Simple recursive Bron-Kerbosch without pivot (n is small)
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

    def check_subdivision(branches):
        k = len(branches)
        if k < 2:
            return None
        
        branch_set = set(branches)
        pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
        
        # Heuristic ordering of pairs: try pairs with shorter shortest paths first?
        # Or just fixed. Let's stick to fixed for now, maybe sort by index sum?
        # Actually, sorting by "difficulty" (longest path) might help backtracking?
        # No, usually easier to assign constrained ones first.
        # Let's just use the natural order.
        
        assigned_paths = {}
        
        def backtrack(idx, used_internal):
            if time.time() > deadline - 0.5:
                return False
            if idx == len(pairs):
                return True
                
            i, j = pairs[idx]
            u = branches[i]
            v = branches[j]
            
            paths_found = []
            
            # DFS to find simple paths from u to v
            # Constraints:
            # - No vertices in used_internal
            # - No vertices in branch_set except u and v
            def dfs_path(curr, target, path, visited):
                if len(paths_found) > 100: # Limit to avoid explosion
                    return
                if curr == target:
                    paths_found.append(path[:])
                    return
                
                for nei in adj_list[curr]:
                    if nei in visited:
                        continue
                    if nei in used_internal:
                        continue
                    if nei in branch_set and nei != target:
                        continue
                    
                    visited.add(nei)
                    path.append(nei)
                    dfs_path(nei, target, path, visited)
                    path.pop()
                    visited.remove(nei)
                    
                    if len(paths_found) > 100:
                        return

            dfs_path(u, v, [u], {u})
            
            if not paths_found:
                return False
            
            # Sort paths by length (prefer shorter to leave room for others)
            paths_found.sort(key=len)
            
            for p in paths_found:
                new_used = set(used_internal)
                for node in p[1:-1]:
                    new_used.add(node)
                
                assigned_paths[(i, j)] = p
                
                if backtrack(idx + 1, new_used):
                    return True
                
                # Backtrack: remove from assigned_paths if we want to be clean, 
                # but since we overwrite on success or return False, it's not strictly 
                # necessary for correctness of the search, only for memory if we retried.
                # We don't retry the same (i,j) with same p, so it's fine.
                
            return False

        if backtrack(0, set()):
            result_paths = []
            for i in range(k):
                for j in range(i+1, k):
                    result_paths.append(assigned_paths[(i, j)])
            return result_paths
        return None

    # Strategy:
    # 1. Start with max clique.
    # 2. Try to increase k by adding vertices.
    # 3. Use local search (swaps) to escape local optima.
    
    deg = [len(adj_list[i]) for i in range(n)]
    sorted_by_deg = sorted(range(n), key=lambda x: -deg[x])
    
    # Helper to try a specific set
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
    # Try to add vertices to the current best set
    current_set = set(best_branches)
    for _ in range(100): # Limit iterations
        if time.time() > deadline - 10.0:
            break
        
        # Find candidates to add
        # Prefer vertices with high degree and many connections to current set
        candidates = []
        for v in range(n):
            if v in current_set:
                continue
            # Score: number of neighbors in current_set + global degree
            score = sum(1 for u in current_set if u in adj[v]) + deg[v]
            candidates.append((score, v))
        
        if not candidates:
            break
            
        candidates.sort(reverse=True)
        
        added = False
        for score, v in candidates[:5]: # Try top 5
            new_set = list(current_set) + [v]
            if try_set(new_set):
                current_set.add(v)
                added = True
                break
        
        if not added:
            # Try swapping: remove one vertex, add two? Or just add one from a different pool?
            # If we can't add, try to swap a low-degree vertex for a high-degree one?
            # Or just break and go to Phase 2.
            break

    # Phase 2: Local Search with Swaps
    # If stuck, try swapping vertices in the current best set to see if we can fit more.
    # Actually, the previous phase increases size. If we can't increase, maybe we can find a DIFFERENT set of the same size that allows expansion?
    # Or just try random sets of size best_k + 1.
    
    # Let's do random sampling of sets of size best_k + 1
    target_k = best_k + 1
    pool_size = min(n, int(target_k * 2.5))
    pool = sorted_by_deg[:pool_size]
    
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 500:
        attempts += 1
        # Sample a set of size target_k from pool
        # Ensure we don't sample the exact same set repeatedly? 
        # random.sample handles uniqueness of elements, but not set uniqueness.
        sample = random.sample(pool, target_k)
        # Sort for consistency (not strictly needed for check_subdivision)
        sample.sort()
        
        if try_set(sample):
            # Found a larger set!
            # Update pool? Maybe keep pool same.
            # Update best_k
            target_k = best_k + 1
            # Reset attempts to allow more tries for the new size
            attempts = 0
            
    # Phase 3: Random restarts with smaller pools or different heuristics?
    # If we still have time, try random sets of size best_k + 1 from the whole graph
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
        # Fallback to clique
        return {
            'branches': [int(b) for b in best_branches],
            'paths': [[int(best_branches[i]), int(best_branches[j])] for i in range(len(best_branches)) for j in range(i+1, len(best_branches))]
        }

# EVOLVE-BLOCK-END
