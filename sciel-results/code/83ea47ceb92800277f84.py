import time
import sys
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_max_clique(adj):
    n = P['n']
    best_clique = []
    
    # Simple Bron-Kerbosch
    def bronk(R, P_set, X_set):
        nonlocal best_clique
        if not P_set and not X_set:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        # Pivot
        U = P_set | X_set
        if U:
            pivot = max(U, key=lambda v: len(adj[v] & P_set))
            candidates = [v for v in P_set if v not in adj[pivot]]
        else:
            candidates = list(P_set)
        
        for v in candidates:
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X_set & adj[v]
            bronk(new_R, new_P, new_X)
            P_set.discard(v)
            X_set.add(v)
            
    bronk(set(), set(range(n)), set())
    return best_clique

def _find_short_paths(adj, u, v, forbidden, max_len=3):
    """Find all simple paths from u to v with length <= max_len, avoiding forbidden vertices."""
    paths = []
    # BFS/DFS up to max_len
    # Since max_len is small, we can just do DFS
    def dfs(curr, path, visited):
        if len(path) - 1 > max_len:
            return
        if curr == v:
            if len(path) > 1: # Must have at least 1 edge
                paths.append(list(path))
            return
        for neighbor in adj[curr]:
            if neighbor not in visited and neighbor not in forbidden:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)
                
    dfs(u, [u], {u})
    return paths

def _check_subdivision(adj, branches, time_limit=2.0):
    """
    Check if a clique subdivision exists for the given set of branches.
    Returns True if found, False otherwise.
    Uses backtracking to find disjoint paths.
    """
    start_time = time.time()
    n = P['n']
    branch_set = set(branches)
    pairs = list(combinations(branches, 2))
    
    # Order pairs: maybe those with fewer available paths first?
    # For simplicity, just use the list as is, or sort by distance?
    # Let's sort by number of initial short paths available (without considering other paths)
    # This is a heuristic to find constrained pairs first.
    
    # Precompute initial path options for each pair ignoring other paths
    # This might be expensive if done naively, but n is small.
    
    def solve(idx, used_interiors):
        if time.time() - start_time > time_limit:
            return False, None # Timeout
        
        if idx == len(pairs):
            return True, None
            
        u, v = pairs[idx]
        forbidden = branch_set - {u, v}
        forbidden.update(used_interiors)
        
        # Find candidate paths
        candidates = _find_short_paths(adj, u, v, forbidden, max_len=3)
        
        # If no candidates, try longer paths?
        if not candidates:
             # Try max_len=4
             candidates = _find_short_paths(adj, u, v, forbidden, max_len=4)
             
        if not candidates:
            return False, None
            
        # Heuristic: try shorter paths first? Or paths that use "less central" vertices?
        # For now, just try all.
        for path in candidates:
            new_used = set(used_interiors)
            new_used.update(path[1:-1]) # Interior vertices
            success, _ = solve(idx + 1, new_used)
            if success:
                return True, None
                
        return False, None

    # To improve performance, we can shuffle pairs or sort them.
    # Let's sort pairs by the number of available short paths (ascending).
    # This requires a pre-pass.
    
    pair_options = []
    for u, v in pairs:
        forbidden = branch_set - {u, v}
        opts = _find_short_paths(adj, u, v, forbidden, max_len=3)
        if not opts:
            opts = _find_short_paths(adj, u, v, forbidden, max_len=4)
        pair_options.append((len(opts), (u, v)))
        
    # If any pair has 0 options, it's impossible
    if any(count == 0 for count, _ in pair_options):
        return False
        
    # Sort pairs by count
    pair_options.sort(key=lambda x: x[0])
    sorted_pairs = [p for _, p in pair_options]
    
    # Re-run solve with sorted pairs
    # We need to modify solve to use sorted_pairs
    def solve_sorted(idx, used_interiors):
        if time.time() - start_time > time_limit:
            return False
        
        if idx == len(sorted_pairs):
            return True
            
        u, v = sorted_pairs[idx]
        forbidden = branch_set - {u, v}
        forbidden.update(used_interiors)
        
        candidates = _find_short_paths(adj, u, v, forbidden, max_len=3)
        if not candidates:
             candidates = _find_short_paths(adj, u, v, forbidden, max_len=4)
             
        if not candidates:
            return False
            
        for path in candidates:
            new_used = set(used_interiors)
            new_used.update(path[1:-1])
            if solve_sorted(idx + 1, new_used):
                return True
                
        return False

    return solve_sorted(0, set())

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    adj = _build_adj()
    
    # 1. Find Max Clique
    clique = _find_max_clique(adj)
    branches = list(clique)
    
    # 2. Try to expand
    # We maintain a set of current branches
    current_branches = set(branches)
    
    # Helper to check if we can add a vertex
    def try_add(v):
        if v in current_branches:
            return False
        new_branches = current_branches | {v}
        # Check subdivision
        if _check_subdivision(adj, list(new_branches), time_limit=5.0):
            current_branches.add(v)
            return True
        return False
        
    def try_swap(u, v):
        if u not in current_branches or v in current_branches:
            return False
        new_branches = (current_branches - {u}) | {v}
        if _check_subdivision(adj, list(new_branches), time_limit=5.0):
            current_branches.remove(u)
            current_branches.add(v)
            return True
        return False

    # Iterative improvement
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        # Try to add any vertex
        for v in range(P['n']):
            if v not in current_branches:
                if try_add(v):
                    improved = True
                    break # Restart loop to re-evaluate with new branch
        if improved:
            continue
            
        # Try to swap
        for u in list(current_branches):
            for v in range(P['n']):
                if v not in current_branches:
                    if try_swap(u, v):
                        improved = True
                        break
            if improved:
                break
                
    # 3. Construct paths for the final set of branches
    # We need to actually find the paths now.
    # We can reuse the logic from _check_subdivision but this time store the paths.
    
    final_branches = list(current_branches)
    branch_set = set(final_branches)
    pairs = list(combinations(final_branches, 2))
    
    # Find paths
    paths = []
    
    # Sort pairs again for determinism and efficiency
    pair_options = []
    for u, v in pairs:
        forbidden = branch_set - {u, v}
        opts = _find_short_paths(adj, u, v, forbidden, max_len=3)
        if not opts:
            opts = _find_short_paths(adj, u, v, forbidden, max_len=4)
        pair_options.append((len(opts), (u, v)))
        
    pair_options.sort(key=lambda x: x[0])
    sorted_pairs = [p for _, p in pair_options]
    
    # Backtracking to find actual paths
    def find_paths(idx, used_interiors, current_paths):
        if time.time() > deadline - 2.0:
            return False
        if idx == len(sorted_pairs):
            return True
            
        u, v = sorted_pairs[idx]
        forbidden = branch_set - {u, v}
        forbidden.update(used_interiors)
        
        candidates = _find_short_paths(adj, u, v, forbidden, max_len=3)
        if not candidates:
             candidates = _find_short_paths(adj, u, v, forbidden, max_len=4)
             
        if not candidates:
            return False
            
        for path in candidates:
            new_used = set(used_interiors)
            new_used.update(path[1:-1])
            current_paths.append(path)
            if find_paths(idx + 1, new_used, current_paths):
                return True
            current_paths.pop()
            
        return False
        
    found_paths = []
    if not find_paths(0, set(), found_paths):
        # Fallback: if the complex search failed (e.g. timeout), 
        # we might have a valid clique but failed to find paths for a larger set.
        # We should revert to the max clique if the larger set failed path finding.
        # But _check_subdivision already verified existence. 
        # If find_paths fails, it might be due to timeout or bug.
        # Let's try to just use the max clique if this fails.
        if len(final_branches) > len(clique):
             final_branches = list(clique)
             branch_set = set(final_branches)
             pairs = list(combinations(final_branches, 2))
             # For clique, paths are direct edges
             found_paths = [[u, v] for u, v in pairs]
        else:
             # Should not happen if _check_subdivision worked
             found_paths = [[u, v] for u, v in pairs] # Fallback to direct edges (invalid if not clique)
             # If this happens, the result might be invalid. 
             # To be safe, if we can't find paths, we should return the max clique result.
             final_branches = list(clique)
             branch_set = set(final_branches)
             pairs = list(combinations(final_branches, 2))
             found_paths = [[u, v] for u, v in pairs]

    # Ensure types
    branches_out = [int(b) for b in final_branches]
    paths_out = [[int(p) for p in path] for path in found_paths]
    
    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
