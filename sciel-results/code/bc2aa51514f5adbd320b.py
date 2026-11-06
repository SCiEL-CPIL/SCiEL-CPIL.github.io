import time
import random

def construct():
    deadline = time.time() + 58  # Safety margin for 60s limit
    
    P_data = {
        'n': 100,
        'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]
    }
    
    n = P_data['n']
    adj = [set() for _ in range(n)]
    for u, v in P_data['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find largest component
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
            
    largest_comp = max(components, key=len)
    comp_size = len(largest_comp)
    
    best_path = []
    
    def greedy_walk(start, strategy):
        """
        strategy: 'warnsdorff', 'random', 'max_deg'
        """
        path = [start]
        visited_set = {start}
        while True:
            current = path[-1]
            neighbors = [v for v in adj[current] if v not in visited_set]
            if not neighbors:
                break
            
            if strategy == 'warnsdorff':
                # Prefer neighbors with fewer unvisited neighbors
                def score(v):
                    return len([u for u in adj[v] if u not in visited_set])
                neighbors.sort(key=score)
                if len(neighbors) > 1:
                    k = min(3, len(neighbors))
                    top_k = neighbors[:k]
                    next_node = random.choice(top_k)
                else:
                    next_node = neighbors[0]
            elif strategy == 'random':
                next_node = random.choice(neighbors)
            elif strategy == 'max_deg':
                # Prefer neighbors with more unvisited neighbors
                def score(v):
                    return len([u for u in adj[v] if u not in visited_set])
                neighbors.sort(key=score, reverse=True)
                next_node = neighbors[0]
            else:
                next_node = random.choice(neighbors)
                
            path.append(next_node)
            visited_set.add(next_node)
            
        return path

    def extend_path(path):
        """Try to extend the path at both ends greedily."""
        path = path[:]
        visited_set = set(path)
        changed = True
        while changed:
            changed = False
            # Extend start
            if path:
                start_node = path[0]
                for nei in adj[start_node]:
                    if nei not in visited_set:
                        path.insert(0, nei)
                        visited_set.add(nei)
                        changed = True
                        break
            # Extend end
            if path:
                end_node = path[-1]
                for nei in adj[end_node]:
                    if nei not in visited_set:
                        path.append(nei)
                        visited_set.add(nei)
                        changed = True
                        break
        return path

    def backtrack_extend(path, max_depth=5, time_limit=1.0):
        """
        Attempt to extend the path from the end using limited backtracking.
        This is more powerful than simple greedy extension as it can undo moves.
        """
        if time.time() > deadline - 2:
            return path
            
        best = path[:]
        visited_set = set(path)
        
        # We will try to extend from the end
        # We allow backtracking up to 'max_depth' steps from the current end
        # This effectively re-optimizes the tail of the path
        
        def dfs(current_path, current_visited, depth):
            nonlocal best, deadline
            
            if time.time() > deadline - 2:
                return current_path
            
            if len(current_path) > len(best):
                best = current_path[:]
                
            if depth == 0:
                return current_path
                
            current_node = current_path[-1]
            neighbors = [v for v in adj[current_node] if v not in current_visited]
            
            if not neighbors:
                return current_path
            
            # Sort neighbors using Warnsdorff's rule to prioritize promising moves
            def score(v):
                return len([u for u in adj[v] if u not in current_visited])
            neighbors.sort(key=score)
            
            best_next = current_path[:]
            
            for nei in neighbors:
                new_path = current_path + [nei]
                new_visited = current_visited | {nei}
                
                # Recurse with reduced depth
                # We don't strictly need to pass depth down if we limit total recursion
                # But for safety, let's just limit total time and branching
                ext = dfs(new_path, new_visited, depth - 1)
                
                if len(ext) > len(best_next):
                    best_next = ext
                    
            return best_next
        
        # Try backtracking from the last few nodes to find a better continuation
        # We take the last 'max_depth' nodes and try to re-route them
        if len(path) > max_depth:
            prefix = path[:-max_depth]
            suffix = path[-max_depth:]
            forbidden = set(prefix)
            
            # Start DFS from the last node of the prefix, but allow it to go into the suffix nodes?
            # Actually, simpler: just run DFS from the current end, but allow it to "undo" by not committing?
            # The standard DFS above only adds. To backtrack, we need to try alternatives at previous steps.
            
            # Let's implement a proper local search:
            # Pick a random point near the end (last 5 nodes) and re-greedy from there?
            # Or, simpler: just run the DFS above which explores all valid extensions.
            # The "backtracking" in the name is slightly misleading if it's just DFS.
            # To truly backtrack, we need to try removing the last node and trying another.
            
            # Let's do a limited beam search or simple backtracking search from the end.
            
            def backtrack_search_from_end(start_path, visited_set, limit_steps=20):
                nonlocal best, deadline
                if time.time() > deadline - 2:
                    return start_path
                
                current = start_path[:]
                vset = visited_set.copy()
                
                # We try to extend. If stuck, we backtrack 1 step and try another neighbor.
                # This is essentially a DFS with pruning.
                
                def dfs_bt(path, vset, depth):
                    nonlocal best, deadline
                    if time.time() > deadline - 2:
                        return path
                    
                    if len(path) > len(best):
                        best = path[:]
                        
                    if depth <= 0:
                        return path
                        
                    node = path[-1]
                    neighbors = [v for v in adj[node] if v not in vset]
                    
                    if not neighbors:
                        return path
                        
                    def score(v):
                        return len([u for u in adj[v] if u not in vset])
                    neighbors.sort(key=score)
                    
                    best_ext = path[:]
                    
                    for nei in neighbors:
                        new_path = path + [nei]
                        new_vset = vset | {nei}
                        ext = dfs_bt(new_path, new_vset, depth - 1)
                        if len(ext) > len(best_ext):
                            best_ext = ext
                            
                    return best_ext
                
                return dfs_bt(current, vset, limit_steps)

        # Apply the backtracking search from the end of the path
        # We limit the search depth to avoid explosion, but allow enough to find improvements
        improved = backtrack_search_from_end(path, visited_set, limit_steps=30)
        
        if len(improved) > len(best):
            best = improved
            
        return best

    strategies = ['warnsdorff', 'random', 'max_deg']
    
    # Phase 1: Generate initial paths
    path_pool = []
    
    while time.time() < deadline - 5:
        for strat in strategies:
            if time.time() >= deadline - 5:
                break
            start = random.choice(largest_comp)
            path = greedy_walk(start, strat)
            path = extend_path(path)
            
            if len(path) > len(best_path):
                best_path = path
            
            if len(path) >= 10:
                path_pool.append(path)
                
            if len(path_pool) > 30:
                path_pool.pop(0)

    # Phase 2: Refine top paths with backtracking
    # Sort pool by length, take top few
    path_pool.sort(key=len, reverse=True)
    
    for path in path_pool[:5]:
        if time.time() >= deadline - 2:
            break
        
        improved = backtrack_extend(path)
        if len(improved) > len(best_path):
            best_path = improved
            best_path = extend_path(best_path)

    # Phase 3: Try merging (if time permits)
    if time.time() < deadline - 1 and len(path_pool) >= 2:
        # Try to merge best_path with any other path in pool if they are disjoint and connected
        best_set = set(best_path)
        for p in path_pool:
            if p == best_path:
                continue
            p_set = set(p)
            if not (best_set & p_set):
                # Check connection
                if best_path[-1] in adj[p[0]]:
                    merged = best_path + p
                    merged = extend_path(merged)
                    if len(merged) > len(best_path):
                        best_path = merged
                elif best_path[-1] in adj[p[-1]]:
                    merged = best_path + p[::-1]
                    merged = extend_path(merged)
                    if len(merged) > len(best_path):
                        best_path = merged
                elif best_path[0] in adj[p[0]]:
                    merged = best_path[::-1] + p
                    merged = extend_path(merged)
                    if len(merged) > len(best_path):
                        best_path = merged
                elif best_path[0] in adj[p[-1]]:
                    merged = best_path[::-1] + p[::-1]
                    merged = extend_path(merged)
                    if len(merged) > len(best_path):
                        best_path = merged

    # Final Validation
    if not best_path:
        best_path = [largest_comp[0]]
        
    # Ensure distinct vertices
    seen = set()
    valid_path = []
    for node in best_path:
        if node not in seen:
            valid_path.append(node)
            seen.add(node)
        else:
            break
    best_path = valid_path
    
    # Ensure valid edges
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in adj[best_path[i]]:
            best_path = best_path[:i+1]
            break
            
    return best_path
