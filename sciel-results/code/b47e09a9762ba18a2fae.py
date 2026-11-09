import time
from collections import deque
from itertools import combinations
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5-second safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency lists and sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    def time_ok():
        return time.time() < deadline

    # Helper: Find all simple paths from s to t up to max_len, avoiding forbidden vertices
    def find_paths_up_to(s, t, forbidden, max_len=4):
        paths = []
        if s in forbidden or t in forbidden:
            return paths
        if s == t:
            return [[s]]
        
        # DFS with length limit
        def dfs(node, path, visited):
            if not time_ok():
                return
            if node == t:
                if len(path) - 1 <= max_len:
                    paths.append(list(path))
                return
            
            if len(path) - 1 >= max_len:
                return
            
            for nb in adj_list[node]:
                if nb == s: # Don't go back to start immediately unless it's a cycle, but simple path means no repeats
                    continue
                if nb in visited:
                    continue
                if nb in forbidden and nb != t:
                    continue
                # Allow going to t even if t is in "forbidden" set of other branches? 
                # No, forbidden contains all branches except s and t. 
                # So t is not in forbidden.
                
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)
        
        visited = {s}
        path = [s]
        dfs(s, path, visited)
        return paths

    # Solver: Find internally disjoint paths for all pairs in 'branches'
    def solve_disjoint_paths(branches, max_path_len=4):
        """
        Returns a list of paths for all pairs in branches, or None if not possible.
        """
        k = len(branches)
        if k < 2:
            return None
        
        branches_set = set(branches)
        pairs = []
        for i in range(k):
            for j in range(i+1, k):
                pairs.append((branches[i], branches[j]))
        
        # Initial state
        used_interior = set()
        # We will use a backtracking search
        # State: (remaining_pairs_index, used_interior, current_paths)
        # To make it efficient, we order pairs dynamically based on constraints.
        
        import sys
        sys.setrecursionlimit(10000)
        
        def backtrack(remaining_pairs, used_interior, current_paths):
            if not time_ok():
                return None
            
            if not remaining_pairs:
                return current_paths
            
            # Heuristic: Choose the pair with the fewest available paths
            min_paths_count = float('inf')
            best_pair_idx = -1
            candidate_paths = []
            
            for i, (u, v) in enumerate(remaining_pairs):
                forbidden = branches_set - {u, v}
                forbidden_with_used = forbidden | used_interior
                paths = find_paths_up_to(u, v, forbidden_with_used, max_path_len)
                if len(paths) < min_paths_count:
                    min_paths_count = len(paths)
                    best_pair_idx = i
                    candidate_paths = paths
                
                if min_paths_count == 0:
                    break
            
            if best_pair_idx == -1 or min_paths_count == 0:
                return None
            
            u, v = remaining_pairs[best_pair_idx]
            new_remaining = [p for i, p in enumerate(remaining_pairs) if i != best_pair_idx]
            
            # Sort candidate paths by length to prefer shorter ones (often better for disjointness)
            candidate_paths.sort(key=len)
            
            for p in candidate_paths:
                interior = set(p[1:-1])
                if interior & used_interior:
                    continue
                
                # Check if interior vertices are distinct from branches (already handled by forbidden)
                # Recurse
                result = backtrack(new_remaining, used_interior | interior, current_paths + [p])
                if result is not None:
                    return result
            
            return None

        # Initial call
        initial_paths = []
        result = backtrack(pairs, set(), initial_paths)
        if result is None:
            return None
        
        # Map paths back to specific pairs to ensure correct ordering for output
        # The result is just a list of paths. We need to associate them with the original pairs.
        # Since we removed pairs one by one, we can reconstruct which path belongs to which pair
        # if we stored that info. 
        # Actually, the backtrack function doesn't track which pair got which path in the final list directly 
        # unless we return a dict. 
        # Let's modify backtrack to return a dict { (u,v): path }
        
        return result # Placeholder, need to fix return type

    # Revised Solver: Returns dict { (u, v): path }
    def solve_disjoint_paths_v2(branches, max_path_len=4):
        k = len(branches)
        if k < 2:
            return None
        
        branches_set = set(branches)
        pairs = []
        for i in range(k):
            for j in range(i+1, k):
                pairs.append((branches[i], branches[j]))
        
        def backtrack(remaining_pairs, used_interior, path_map):
            if not time_ok():
                return None
            
            if not remaining_pairs:
                return path_map
            
            # Heuristic: MRV (Minimum Remaining Values)
            min_paths_count = float('inf')
            best_pair_idx = -1
            candidate_paths = []
            
            for i, (u, v) in enumerate(remaining_pairs):
                forbidden = branches_set - {u, v}
                forbidden_with_used = forbidden | used_interior
                paths = find_paths_up_to(u, v, forbidden_with_used, max_path_len)
                if len(paths) < min_paths_count:
                    min_paths_count = len(paths)
                    best_pair_idx = i
                    candidate_paths = paths
                
                if min_paths_count == 0:
                    break
            
            if best_pair_idx == -1 or min_paths_count == 0:
                return None
            
            u, v = remaining_pairs[best_pair_idx]
            new_remaining = [p for i, p in enumerate(remaining_pairs) if i != best_pair_idx]
            
            candidate_paths.sort(key=len)
            
            for p in candidate_paths:
                interior = set(p[1:-1])
                if interior & used_interior:
                    continue
                
                new_path_map = dict(path_map)
                new_path_map[(u, v)] = p
                
                result = backtrack(new_remaining, used_interior | interior, new_path_map)
                if result is not None:
                    return result
            
            return None

        result = backtrack(pairs, set(), {})
        return result

    # Find a good initial clique
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
            candidates.sort(key=lambda x: -len(adj[x]))
            
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

    # Try the max clique first
    if len(max_clique) >= 2:
        # Try with increasing path length if needed, start with short paths for speed
        for mlen in [3, 4, 5]:
            if not time_ok():
                break
            paths_dict = solve_disjoint_paths_v2(max_clique, mlen)
            if paths_dict:
                best_result = (max_clique, paths_dict)
                best_k = len(max_clique)
                break

    # If full clique failed, try subsets or use local search to grow from a smaller valid set
    if best_result is None or best_k < len(max_clique):
        # Start with a small valid set, e.g., an edge
        # Find an edge
        init_branches = None
        for u, v in edges:
            if not time_ok():
                break
            paths_dict = solve_disjoint_paths_v2([u, v], 3)
            if paths_dict:
                init_branches = [u, v]
                break
        
        if init_branches is None:
            # Fallback to first edge
            init_branches = [edges[0][0], edges[0][1]]
            paths_dict = solve_disjoint_paths_v2(init_branches, 3)
            if paths_dict is None:
                # Should not happen for an edge
                return {'branches': [0, 1], 'paths': [[0, 1]]}
            
        best_result = (init_branches, paths_dict)
        best_k = len(init_branches)

    # Local Search: Try to add vertices
    current_branches = list(best_result[0])
    current_paths = best_result[1]
    
    improved = True
    while improved and time_ok():
        improved = False
        # Try adding each non-branch vertex
        candidates = [v for v in range(n) if v not in current_branches]
        random.shuffle(candidates)
        
        for v in candidates:
            if not time_ok():
                break
            
            new_branches = sorted(current_branches + [v])
            # Try to solve for the new set
            # Use a slightly larger path length limit as the set grows
            mlen = 4 if len(new_branches) <= 5 else 5
            new_paths = solve_disjoint_paths_v2(new_branches, mlen)
            
            if new_paths:
                current_branches = new_branches
                current_paths = new_paths
                best_result = (current_branches, current_paths)
                best_k = len(current_branches)
                improved = True
                # Restart the loop to try adding another vertex
                break

    # Local Search: Try swapping to escape local optima
    # If we are stuck, try removing a branch and adding two others? 
    # Or just try to add again with different random order.
    if time_ok():
        for _ in range(10): # A few more rounds
            if not time_ok():
                break
            improved = False
            candidates = [v for v in range(n) if v not in current_branches]
            random.shuffle(candidates)
            for v in candidates:
                if not time_ok():
                    break
                new_branches = sorted(current_branches + [v])
                new_paths = solve_disjoint_paths_v2(new_branches, 5)
                if new_paths:
                    current_branches = new_branches
                    current_paths = new_paths
                    best_result = (current_branches, current_paths)
                    best_k = len(current_branches)
                    improved = True
                    break
            if not improved:
                break

    # Construct final output
    branches_out = [int(b) for b in current_branches]
    paths_out = []
    k = len(branches_out)
    for i in range(k):
        for j in range(i+1, k):
            u, v = branches_out[i], branches_out[j]
            # The key in current_paths is (u, v) where u < v? 
            # In solve_disjoint_paths_v2, we added pairs as (branches[i], branches[j]) with i<j.
            # So the key is (min, max).
            key = (min(u, v), max(u, v))
            p = current_paths[key]
            # Ensure path starts with u and ends with v
            if p[0] == u and p[-1] == v:
                paths_out.append([int(x) for x in p])
            elif p[0] == v and p[-1] == u:
                paths_out.append([int(x) for x in reversed(p)])
            else:
                # Should not happen
                paths_out.append([int(x) for x in p])

    return {
        'branches': branches_out,
        'paths': paths_out
    }

# Run construct to test
# print(construct())
