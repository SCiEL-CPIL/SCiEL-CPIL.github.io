import time
import sys
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    n = P['n']
    adj_list = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)
    
    def time_check():
        return time.time() > deadline

    def find_shortest_path(src, dst, blocked):
        """Find a simple path from src to dst avoiding blocked vertices (except src and dst).
        Returns the path as a list of vertices, or None."""
        if src == dst:
            return [src]
        if src in blocked or dst in blocked:
            # src and dst are allowed to be in blocked for the check logic, 
            # but we must ensure we don't step on other blocked vertices.
            pass
        
        parent = {src: None}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for neighbor in adj_list[node]:
                if neighbor in parent:
                    continue
                if neighbor in blocked and neighbor != dst:
                    continue
                parent[neighbor] = node
                queue.append(neighbor)
        return None

    def check_feasibility(branches):
        """Check if the given set of branches supports a clique subdivision.
        Returns the paths dict if feasible, else None."""
        if len(branches) < 2:
            return None
        
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Pre-compute options for each pair
        # Option: 'direct' if edge exists, else set of common neighbors (for length 2 paths)
        # We will use backtracking to assign paths.
        
        pair_data = {}
        non_direct_pairs = []
        for u, v in pairs:
            if u in adj_set[v]:
                pair_data[(u, v)] = {'type': 'direct', 'options': None}
            else:
                common = set()
                for w in adj_list[u]:
                    if w not in branch_set and w in adj_set[v]:
                        common.add(w)
                pair_data[(u, v)] = {'type': 'indirect', 'options': common}
                non_direct_pairs.append((u, v))
        
        # If any non-direct pair has no common neighbor, it might still be feasible via longer paths.
        # But for initial pruning, if no common neighbor, we mark it as "needs long path".
        
        # Sort non-direct pairs by number of options (most constrained first)
        non_direct_pairs.sort(key=lambda p: len(pair_data[p]['options']))
        
        assignment = {}
        used_internals = set()
        
        def backtrack(idx):
            if time_check():
                return False
            if idx == len(non_direct_pairs):
                return True
            
            u, v = non_direct_pairs[idx]
            
            # We need to find a path from u to v that avoids used_internals and other branches.
            # The path can be of any length >= 2.
            # To keep it fast, we first try length 2 paths (using common neighbors).
            # If that fails, we try to find any path via BFS.
            
            blocked = branch_set | used_internals
            blocked.discard(u)
            blocked.discard(v)
            
            # Try length 2 paths first (faster to enumerate and assign)
            common_neighbors = pair_data[(u, v)]['options']
            found_path = None
            
            # Filter common neighbors that are not used
            available_commons = [w for w in common_neighbors if w not in used_internals]
            
            # Randomize order to avoid bias? Or keep deterministic?
            # For now, deterministic.
            for w in available_commons:
                # Path is [u, w, v]
                # Check if w is valid (not branch, not used)
                # w is not in branch_set by construction of common_neighbors
                # w is not in used_internals by filter
                found_path = [u, w, v]
                break
            
            if found_path is None:
                # Try to find any path via BFS
                found_path = find_shortest_path(u, v, blocked)
            
            if found_path is None:
                return False
            
            internal = found_path[1:-1]
            
            # Check if internal vertices are valid (disjoint, not branches)
            for w in internal:
                if w in branch_set or w in used_internals:
                    return False
            
            # Assign
            for w in internal:
                used_internals.add(w)
            assignment[(u, v)] = internal
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            for w in internal:
                used_internals.discard(w)
            del assignment[(u, v)]
            
            return False

        if not backtrack(0):
            return None
        
        # Construct final paths
        result_paths = {}
        for pair in pairs:
            u, v = pair
            if pair_data[pair]['type'] == 'direct':
                result_paths[pair] = [u, v]
            else:
                internal = assignment[pair]
                result_paths[pair] = [u] + internal + [v]
        
        return result_paths

    def solve():
        # Sort vertices by degree descending
        degree = [len(adj_list[i]) for i in range(n)]
        sorted_vertices = sorted(range(n), key=lambda x: -degree[x])
        
        best_result = None
        best_k = 0
        
        # Iterate k from n down to 2
        for k in range(n, 1, -1):
            # Find a subset of size k that is feasible
            # Use backtracking to build the subset
            current_branches = []
            
            def backtrack_subset(idx):
                if time_check():
                    return False
                
                if len(current_branches) == k:
                    # Check feasibility
                    paths = check_feasibility(current_branches)
                    if paths is not None:
                        # Found a valid set of size k
                        return True
                    return False
                
                # Pruning: if we can't reach size k
                if len(current_branches) + (n - idx) < k:
                    return False
                
                for i in range(idx, n):
                    if time_check():
                        return False
                    
                    candidate = sorted_vertices[i]
                    
                    # Add candidate
                    current_branches.append(candidate)
                    
                    # Optional: Check if current partial set is "feasible" in a weaker sense?
                    # For now, we only check when full size is reached to save time.
                    # However, checking earlier can prune more.
                    # Let's check if the current set is feasible (subset of a valid set).
                    # If the current set is NOT feasible, no superset can be feasible.
                    # This is a strong pruning step.
                    if len(current_branches) >= 2:
                         if check_feasibility(current_branches) is None:
                             current_branches.pop()
                             continue
                    
                    if backtrack_subset(i + 1):
                        return True
                    
                    current_branches.pop()
                
                return False
            
            if backtrack_subset(0):
                # We found a valid set of size k
                paths = check_feasibility(current_branches)
                # paths is not None
                best_result = {
                    'branches': [int(x) for x in current_branches],
                    'paths': [[int(x) for x in p] for p in sorted(paths.values(), key=lambda p: (p[0], p[-1]))]
                }
                best_k = k
                break # Since we iterate k downwards, first found is max
                
        if best_result is not None:
            return best_result
        
        # Fallback
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
        return {'branches': [0, 1], 'paths': [[0, 1]]}

    result = solve()
    
    if result is not None:
        result['branches'] = [int(x) for x in result['branches']]
        result['paths'] = [[int(x) for x in p] for p in result['paths']]
    
    return result
# EVOLVE-BLOCK-END
