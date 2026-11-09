import time
import collections
import random
import itertools

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = P['n']
    edges = P['edges']
    adj = collections.defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # 1. Find Max Clique to establish a lower bound
    def find_max_clique():
        max_clique = []
        candidates = list(range(n))
        
        def expand(clique, candidates):
            nonlocal max_clique
            if len(clique) > len(max_clique):
                max_clique = clique[:]
            if not candidates:
                return
            # Bron-Kerbosch with pivot is better, but for n=24 simple is ok
            # Simple recursive
            for i in range(len(candidates)):
                v = candidates[i]
                new_candidates = [u for u in candidates[i+1:] if u in adj[v]]
                expand(clique + [v], new_candidates)
                
        expand([], candidates)
        return max_clique

    max_clique = find_max_clique()
    best_k = len(max_clique)
    best_branches = max_clique
    best_paths = [[max_clique[i], max_clique[j]] for i in range(best_k) for j in range(i+1, best_k)]
    
    # Helper to check if a set of branches forms a valid subdivision
    # Returns list of paths if valid, else None
    def check_subdivision(branches):
        k = len(branches)
        if k < 2:
            return None
        
        branch_set = set(branches)
        pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
        
        # Order pairs to process. 
        # Heuristic: Pairs with shorter shortest path distance first? 
        # Or just fixed order. Let's use fixed order but maybe sort by distance in original graph.
        # We need distance in the graph avoiding other branches? That's complex.
        # Let's just use the order.
        
        # Backtracking state
        used_internal = set()
        assigned_paths = {}
        
        # Recursive search
        def backtrack(idx):
            if time.time() > deadline - 0.5: # Safety margin
                return False
            if idx == len(pairs):
                return True
                
            i, j = pairs[idx]
            u = branches[i]
            v = branches[j]
            
            # Find all simple paths from u to v
            # Constraints:
            # 1. Cannot use vertices in used_internal
            # 2. Cannot use vertices in branch_set except u and v
            # 3. Must be simple
            
            # We use DFS to enumerate paths. 
            # To keep it fast, we limit the number of paths we consider or the depth.
            # For small k, paths should be short.
            
            # Optimization: If u and v are adjacent, the direct edge is a candidate.
            # We should prioritize shorter paths.
            
            paths_found = []
            
            # DFS with pruning
            def dfs_path(curr, target, path, visited):
                if len(paths_found) > 20: # Limit candidates to avoid explosion
                    return
                if curr == target:
                    paths_found.append(path[:])
                    return
                
                # Pruning: if path is already longer than some heuristic limit?
                # Let's just rely on the 20 limit and time.
                
                for nei in adj[curr]:
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
                    
                    if len(paths_found) > 20:
                        return

            dfs_path(u, v, [u], {u})
            
            if not paths_found:
                return False
            
            # Sort paths by length
            paths_found.sort(key=len)
            
            # Try each path
            for p in paths_found:
                # Mark internal vertices
                new_used = set(used_internal)
                for node in p[1:-1]:
                    new_used.add(node)
                
                # Check if new_used causes conflict? 
                # We just pass it down.
                
                assigned_paths[(i, j)] = p
                # Save state of used_internal? 
                # We need to restore it. 
                # Since used_internal is a set, we can just copy it or use a stack.
                # Copying is fine for small k.
                
                # Wait, `used_internal` is modified in the outer scope if we don't pass it.
                # Let's pass it as an argument to backtrack to keep it pure.
                # Refactoring backtrack to take used_internal as arg.
                
                # Actually, the current structure uses closure variable `used_internal`.
                # This makes backtracking tricky with sets.
                # Let's refactor `backtrack` to accept `used_internal`.
                
                pass # Placeholder for refactoring logic below

            return False

        # Refactored Backtrack
        def backtrack_v2(idx, used_internal):
            if time.time() > deadline - 0.5:
                return False
            if idx == len(pairs):
                return True
                
            i, j = pairs[idx]
            u = branches[i]
            v = branches[j]
            
            paths_found = []
            
            def dfs_path(curr, target, path, visited):
                if len(paths_found) > 50: # Increased limit slightly
                    return
                if curr == target:
                    paths_found.append(path[:])
                    return
                
                for nei in adj[curr]:
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
                    
                    if len(paths_found) > 50:
                        return

            dfs_path(u, v, [u], {u})
            
            if not paths_found:
                return False
            
            paths_found.sort(key=len)
            
            for p in paths_found:
                new_used = set(used_internal)
                for node in p[1:-1]:
                    new_used.add(node)
                
                assigned_paths[(i, j)] = p
                
                if backtrack_v2(idx + 1, new_used):
                    return True
                
                # Backtrack
                # No need to explicitly remove from assigned_paths if we just overwrite or ignore on fail
                # But we should clean up if we want to try next path?
                # assigned_paths is overwritten in the next iteration of the loop.
                # The `new_used` is local to this iteration, so no state leak there.
                
            return False

        if backtrack_v2(0, set()):
            result_paths = []
            for i in range(k):
                for j in range(i+1, k):
                    result_paths.append(assigned_paths[(i, j)])
            return result_paths
        return None

    # Strategy:
    # 1. We have a max clique of size C.
    # 2. Try to find a set of size C+1, C+2, ...
    # 3. Candidate generation:
    #    a. Top k vertices by degree.
    #    b. Max clique + one random neighbor.
    #    c. Random sets of size k from top 2k degree vertices.
    
    deg = [len(adj[i]) for i in range(n)]
    sorted_by_deg = sorted(range(n), key=lambda x: -deg[x])
    
    # Pre-compute a few good candidate sets
    # We will iterate k from best_k + 1 upwards
    # To avoid missing the optimum, we might also try to find a better set of size best_k
    # but the clique is likely optimal for small sizes.
    
    # Let's try to increase k
    found_better = False
    for k in range(best_k + 1, min(n, 10) + 1): # Limit to 10 for safety
        if time.time() > deadline - 5.0:
            break
            
        candidates = set()
        
        # 1. Top k degree
        candidates.add(tuple(sorted(sorted_by_deg[:k])))
        
        # 2. Max clique extended
        # Add one vertex from neighbors of max clique
        clique_set = set(max_clique)
        neighbors = set()
        for v in max_clique:
            for u in adj[v]:
                if u not in clique_set:
                    neighbors.add(u)
        # Try adding each neighbor
        for nb in list(neighbors)[:20]: # Limit iterations
            cand = list(clique_set) + [nb]
            if len(cand) == k:
                candidates.add(tuple(sorted(cand)))
            elif len(cand) < k:
                # Fill with top degree vertices not in cand
                rest = [v for v in sorted_by_deg if v not in cand]
                while len(cand) < k and rest:
                    cand.append(rest.pop(0))
                if len(cand) == k:
                    candidates.add(tuple(sorted(cand)))
        
        # 3. Random samples from top 1.5k
        pool_size = int(k * 1.5) + 1
        pool = sorted_by_deg[:pool_size]
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            sample = random.sample(pool, k)
            candidates.add(tuple(sorted(sample)))
            
        for cand in candidates:
            if time.time() > deadline - 2.0:
                break
            paths = check_subdivision(list(cand))
            if paths:
                best_k = k
                best_branches = list(cand)
                best_paths = paths
                found_better = True
                # If we found a valid set of size k, we can break to try k+1
                # But maybe another candidate of size k is "better"? No, score is just k.
                # So we can break to increase k.
                break
        
        if not found_better:
            # If we couldn't find size k, it's unlikely we find k+1?
            # Not necessarily, but usually yes.
            # We can continue, but with time pressure.
            pass

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
