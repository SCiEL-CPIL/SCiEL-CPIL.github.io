import time
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
    
    # Sort by size descending
    components.sort(key=len, reverse=True)
    
    # We only care about the largest component(s)
    # For a sparse random graph G(100, 0.018), the largest component is likely most of the vertices
    
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin for 90s limit
    
    best_path = []
    best_len = 0
    
    # Warnsdorff-guided DFS with backtracking
    # The key insight: in sparse graphs, greedy gets stuck. Backtracking lets us explore
    # alternative branches when we hit a dead end.
    
    def dfs_warnsdorff(start, max_depth, time_limit):
        """DFS with Warnsdorff heuristic and backtracking to find longest path."""
        nonlocal best_path, best_len
        
        path = [start]
        visited_set = {start}
        # Precompute adjacency as sorted lists for efficiency
        adj_list = [sorted(adj[i]) for i in range(n)]
        
        def extend():
            nonlocal best_path, best_len
            current_path = path
            current_visited = visited_set
            
            while len(current_path) < max_depth:
                last = current_path[-1]
                candidates = [v for v in adj_list[last] if v not in current_visited]
                
                if not candidates:
                    break
                
                # Warnsdorff: sort by number of unvisited neighbors (ascending)
                # This tends to keep options open
                candidates.sort(key=lambda v: (len([u for u in adj_list[v] if u not in current_visited]), v))
                
                # Try each candidate in order
                found = False
                for v in candidates:
                    current_path.append(v)
                    current_visited.add(v)
                    
                    if len(current_path) > best_len:
                        best_len = len(current_path)
                        best_path = list(current_path)
                    
                    # Recurse
                    if len(current_path) < max_depth:
                        # Continue extending from v
                        sub_found = extend_from(current_path, current_visited, max_depth, time_limit)
                        if sub_found:
                            found = True
                    
                    # Check time
                    if time.time() > time_limit:
                        return False
                    
                    # Backtrack
                    current_path.pop()
                    current_visited.discard(v)
                
                if found:
                    # We found a better path deeper, backtrack to let caller handle
                    # Actually, we should keep exploring
                    pass
                
                # If we exhausted all candidates from last, we need to backtrack
                # This is handled by the caller
                if not candidates or all(v in current_visited for v in adj_list[last]):
                    break
            return True
        
        def extend_from(current_path, current_visited, max_depth, time_limit):
            """Recursive extension with backtracking."""
            nonlocal best_path, best_len
            
            if time.time() > time_limit:
                return False
            
            last = current_path[-1]
            candidates = [v for v in adj_list[last] if v not in current_visited]
            
            if not candidates:
                return False
            
            # Warnsdorff ordering
            candidates.sort(key=lambda v: (len([u for u in adj_list[v] if u not in current_visited]), v))
            
            for v in candidates:
                if time.time() > time_limit:
                    return False
                
                current_path.append(v)
                current_visited.add(v)
                
                if len(current_path) > best_len:
                    best_len = len(current_path)
                    best_path = list(current_path)
                
                if len(current_path) < max_depth:
                    extend_from(current_path, current_visited, max_depth, time_limit)
                
                # Backtrack
                current_path.pop()
                current_visited.discard(v)
            
            return True
        
        # Start the search
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        if len(path) < max_depth:
            extend_from(path, visited_set, max_depth, time_limit)
        
        return best_path
    
    # Phase 1: Try DFS from each vertex in the largest component
    largest_comp = components[0] if components else []
    
    # Try from each vertex, with increasing time budgets
    # Start with shorter searches to get a baseline, then longer ones
    for start in largest_comp:
        if time.time() > deadline:
            break
        
        # Give each start a reasonable time budget
        remaining = deadline - time.time()
        if remaining < 0.5:
            break
        
        # Allocate time based on how many starts we've done
        n_starts = len(largest_comp)
        time_per_start = max(0.1, remaining / max(1, n_starts - largest_comp.index(start)))
        
        dfs_warnsdorff(start, n, start_time + time.time() - start_time + min(time_per_start, 5.0))
    
    # Phase 2: If we have time, do more targeted searches
    if time.time() < deadline and best_len < n:
        # Try from vertices not yet in the best path
        in_best = set(best_path)
        for start in largest_comp:
            if time.time() > deadline:
                break
            if start in in_best:
                continue
            
            remaining = deadline - time.time()
            if remaining < 0.5:
                break
            
            time_budget = min(remaining, 3.0)
            dfs_warnsdorff(start, n, time.time() + time_budget)
    
    # Phase 3: Final attempt with all remaining time
    if time.time() < deadline and best_len < n:
        # Try the best remaining starts with full time
        remaining = deadline - time.time()
        if remaining > 1.0:
            # Pick a few promising starts
            in_best = set(best_path)
            candidates = [v for v in largest_comp if v not in in_best]
            if not candidates:
                candidates = largest_comp
            
            # Try each with a portion of remaining time
            n_cand = min(len(candidates), max(1, int(remaining / 2)))
            for start in candidates[:n_cand]:
                if time.time() > deadline:
                    break
                time_budget = min(deadline - time.time(), 2.0)
                if time_budget < 0.1:
                    break
                dfs_warnsdorff(start, n, time.time() + time_budget)
    
    # Ensure we return a valid non-empty path
    if not best_path:
        # Fallback: return any single vertex
        if components:
            best_path = [components[0][0]]
        else:
            best_path = [0]
    
    return best_path
