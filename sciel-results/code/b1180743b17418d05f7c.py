import time
from itertools import combinations
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t0 = time.time()
    HARD_DEADLINE = t0 + 85.0  # Leave 5s margin for evaluator's 90s kill
    
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for e in edges:
        a, b = e
        edge_set.add((a, b))
        edge_set.add((b, a))
    adj = [set() for _ in range(n)]
    for e in edges:
        a, b = e
        adj[a].add(b)
        adj[b].add(a)
    
    def time_up():
        return time.time() > HARD_DEADLINE
    
    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    def bfs_shortest_paths(s, t, blocked, max_paths=10, max_len=12):
        """Find up to max_paths simple paths from s to t avoiding blocked (except s,t).
        Uses BFS to find shortest paths first, then extends."""
        if s == t:
            return [[s]]
        
        # First, find all shortest paths via BFS
        # Then find slightly longer paths via DFS with depth limit
        
        # BFS to find shortest path length
        from collections import deque
        dist = {s: 0}
        parent = {s: None}
        q = deque([s])
        found_t = False
        while q:
            curr = q.popleft()
            if curr == t:
                found_t = True
                break
            for nb in sorted(adj[curr]):
                if nb not in dist:
                    if nb in blocked and nb != s and nb != t:
                        continue
                    dist[nb] = dist[curr] + 1
                    parent[nb] = curr
                    q.append(nb)
        
        if not found_t:
            return []
        
        shortest_len = dist[t]
        
        # Find all paths of length shortest_len using DFS
        paths = []
        
        def dfs_all_paths(curr, path, visited, max_depth):
            if len(paths) >= max_paths:
                return
            if curr == t:
                if len(path) <= max_len + 2:
                    paths.append(list(path))
                return
            if len(path) - 1 > max_depth:
                return
            for nb in sorted(adj[curr]):
                if nb in visited:
                    continue
                if nb in blocked and nb != s and nb != t:
                    continue
                visited.add(nb)
                path.append(nb)
                dfs_all_paths(nb, path, visited, max_depth)
                path.pop()
                visited.remove(nb)
                if len(paths) >= max_paths:
                    return
        
        dfs_all_paths(s, [s], {s}, max_len)
        
        if not paths:
            return []
        
        # If we have fewer than max_paths, try to find more with longer paths
        if len(paths) < max_paths:
            existing_internal = set()
            for p in paths:
                existing_internal.update(p[1:-1])
            
            def dfs_more_paths(curr, path, visited, max_depth):
                if len(paths) >= max_paths:
                    return
                if curr == t:
                    if len(path) <= max_len + 2:
                        internal = set(path[1:-1])
                        if not (internal & existing_internal):
                            paths.append(list(path))
                            existing_internal.update(internal)
                    return
                if len(path) - 1 > max_depth:
                    return
                for nb in sorted(adj[curr]):
                    if nb in visited:
                        continue
                    if nb in blocked and nb != s and nb != t:
                        continue
                    visited.add(nb)
                    path.append(nb)
                    dfs_more_paths(nb, path, visited, max_depth)
                    path.pop()
                    visited.remove(nb)
                    if len(paths) >= max_paths:
                        return
            
            dfs_more_paths(s, [s], {s}, max_len)
        
        return paths
    
    def solve_for_branches(branches, max_paths_per_pair=5, max_path_len=10):
        """
        Solve for a given set of branches. Returns list of paths or None.
        Uses backtracking with most-constrained-first ordering.
        """
        bs = set(branches)
        sorted_bs = sorted(branches)
        pairs = list(combinations(sorted_bs, 2))
        
        # Order pairs by "constrainedness": fewer common neighbors = more constrained
        def pair_score(p):
            u, v = p
            common = adj[u] & adj[v] - bs
            return len(common)
        
        # Most constrained first (fewest options)
        ordered_pairs = sorted(pairs, key=pair_score)
        
        solution_paths = [None] * len(ordered_pairs)
        
        def solve(idx, used_internal):
            if time_up():
                return False
            if idx == len(ordered_pairs):
                return True
            
            u, v = ordered_pairs[idx]
            blocked = used_internal | bs
            
            # Find candidate paths
            candidates = bfs_shortest_paths(u, v, blocked, 
                                            max_paths=max_paths_per_pair, 
                                            max_len=max_path_len)
            
            if not candidates:
                return False
            
            # Sort by length (shortest first)
            candidates.sort(key=len)
            
            for path in candidates:
                internal = set(path[1:-1])
                if internal & used_internal:
                    continue
                if internal & bs:
                    continue
                
                solution_paths[idx] = path
                for v_node in internal:
                    used_internal.add(v_node)
                
                if solve(idx + 1, used_internal):
                    return True
                
                for v_node in internal:
                    used_internal.discard(v_node)
                solution_paths[idx] = None
            
            return False
        
        if solve(0, set()):
            # Reorder to standard combination order
            final_paths = {}
            for i, pair in enumerate(ordered_pairs):
                final_paths[pair] = solution_paths[i]
            
            result = []
            for p in pairs:
                if p in final_paths:
                    result.append(final_paths[p])
                else:
                    return None
            return result
        return None
    
    # Start with a 2-branch solution
    u, v = edges[0]
    best = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    best_size = 2
    best_branches = {u, v}
    
    vertices = list(range(n))
    deg_sorted = sorted(vertices, key=lambda x: len(adj[x]), reverse=True)
    
    # Try to grow incrementally
    for target_size in range(3, min(n + 1, 15)):
        if time_up():
            break
        
        if target_size <= best_size:
            continue
        
        # Candidate vertices to try adding
        candidates = [v for v in deg_sorted if v not in best_branches]
        
        # Try each candidate
        found = False
        for cand in candidates[:min(len(candidates), 15)]:
            if time_up():
                break
            
            new_branches = best_branches | {cand}
            new_branches_list = sorted(new_branches)
            
            paths = solve_for_branches(new_branches_list, 
                                       max_paths_per_pair=5, 
                                       max_path_len=10)
            if paths is not None:
                best = {'branches': [int(x) for x in new_branches_list], 
                        'paths': [[int(x) for x in p] for p in paths]}
                best_size = target_size
                best_branches = new_branches
                found = True
                break
        
        if not found:
            # Try random combinations of new vertices
            remaining = [v for v in deg_sorted if v not in best_branches]
            random.seed(42 + target_size)
            for _ in range(10):
                if time_up():
                    break
                if len(remaining) < target_size - best_size:
                    break
                to_add = random.sample(remaining, target_size - best_size)
                new_branches = best_branches | set(to_add)
                new_branches_list = sorted(new_branches)
                
                paths = solve_for_branches(new_branches_list,
                                           max_paths_per_pair=5,
                                           max_path_len=10)
                if paths is not None:
                    best = {'branches': [int(x) for x in new_branches_list],
                            'paths': [[int(x) for x in p] for p in paths]}
                    best_size = target_size
                    best_branches = new_branches
                    found = True
                    break
        
        if not found:
            # Try larger jumps
            if target_size - best_size > 1:
                continue
            else:
                # Can't grow anymore with current strategy
                # Try completely different subsets
                break
    
    # If we didn't find anything better, try random subsets
    if best_size == 2:
        random.seed(123)
        for k in range(3, min(n + 1, 12)):
            if time_up():
                break
            for _ in range(20):
                if time_up():
                    break
                subset = random.sample(vertices, k)
                paths = solve_for_branches(sorted(subset),
                                           max_paths_per_pair=5,
                                           max_path_len=10)
                if paths is not None:
                    best = {'branches': [int(x) for x in sorted(subset)],
                            'paths': [[int(x) for x in p] for p in paths]}
                    best_size = k
                    break
    
    if best is None:
        return fallback()
    
    return best
