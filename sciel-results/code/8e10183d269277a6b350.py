import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
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
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        queue.append(nei)
            components.append(comp)
            
    # Sort components by size, largest first
    components.sort(key=len, reverse=True)
    
    if not components:
        return [0]
        
    best_path = []
    
    # We focus on the largest component. 
    # If the largest component is small enough, we can try harder.
    # For each component, we try to find the longest path.
    # Given time constraints, we prioritize the largest one.
    
    deadline = time.time() + 85  # 5s safety margin
    
    def find_longest_path_in_comp(comp):
        if not comp:
            return []
        comp_set = set(comp)
        # Map nodes to 0..m-1 for internal processing? 
        # Or just use original node IDs.
        
        # Greedy initial path
        def greedy_path(start):
            path = [start]
            seen = {start}
            while True:
                last = path[-1]
                candidates = [v for v in adj[last] if v not in seen and v in comp_set]
                if not candidates:
                    break
                # Warnsdorff: pick candidate with fewest unvisited neighbors
                candidates.sort(key=lambda v: (len([u for u in adj[v] if u not in seen and u in comp_set]), v))
                path.append(candidates[0])
                seen.add(candidates[0])
            return path
        
        best_greedy = []
        for node in comp:
            p = greedy_path(node)
            if len(p) > len(best_greedy):
                best_greedy = p
                
        current_best = best_greedy
        if not current_best:
            return comp[:1]
            
        target_len = len(current_best) + 1
        max_len = len(comp)
        
        # IDDFS
        # We try to find a path of length `target_len`
        # If found, update best and increment target_len
        # If not found, return best
        
        # To speed up, we can use a heuristic search (IDA*) or just DFS with pruning.
        # DFS with pruning:
        # State: current_node, visited_set, path
        # Prune if len(path) + (max possible extension) < target_len
        
        # Helper: get unvisited neighbors in comp
        def get_unvisited_neighbors(node, visited_set):
            return [v for v in adj[node] if v not in visited_set and v in comp_set]
        
        # Upper bound on extension from a node:
        # The number of unvisited nodes reachable from `node` in the graph induced by `unvisited`.
        # This is expensive to compute at every step.
        # Approximation: count of unvisited neighbors + ...
        # Simpler: if len(path) + (number of unvisited nodes in comp) < target_len, prune.
        # This is weak.
        
        # Better pruning: 
        # If the current node has no unvisited neighbors, and len(path) < target_len, fail.
        
        # Let's use a recursive DFS with a step limit or time check.
        # To avoid recursion depth issues, we can increase limit, but 100 is fine.
        
        found = [False]
        best_path_found = [current_best]
        
        def dfs(node, visited_set, path):
            nonlocal found
            if time.time() > deadline:
                return False
            
            if len(path) == target_len:
                found[0] = True
                best_path_found[0] = path[:]
                return True
            
            # Pruning
            # If we can't reach target_len even if we visit all remaining reachable nodes
            # Simple bound: len(path) + (len(comp_set) - len(visited_set)) < target_len
            if len(path) + (len(comp_set) - len(visited_set)) < target_len:
                return False
                
            neighbors = get_unvisited_neighbors(node, visited_set)
            if not neighbors:
                return False
                
            # Sort neighbors by Warnsdorff heuristic
            # Fewer unvisited neighbors first
            def heuristic(v):
                return len([u for u in adj[v] if u not in visited_set and u in comp_set])
            
            neighbors.sort(key=heuristic)
            
            for nei in neighbors:
                new_visited = visited_set | {nei}
                new_path = path + [nei]
                if dfs(nei, new_visited, new_path):
                    return True
            return False
        
        # Try to increase path length
        while target_len <= max_len and time.time() < deadline:
            found[0] = False
            # We can restart from different starts? 
            # IDDFS usually fixes the start or tries all starts.
            # Trying all starts is expensive.
            # However, for longest path, the start doesn't matter much if we can traverse back and forth? 
            # No, it's a simple path.
            # We can try starting from each node in comp? 
            # That's 100 * (search cost).
            
            # Optimization: Only try starts that are likely to be part of a long path.
            # Or, just try one start (e.g., the one from greedy) and if it fails, try others?
            # Actually, a path of length L can start anywhere.
            # Let's try starting from each node in comp, but break early if found.
            
            success = False
            for start in comp:
                if time.time() > deadline:
                    break
                visited_init = {start}
                path_init = [start]
                if dfs(start, visited_init, path_init):
                    success = True
                    break
            
            if success:
                current_best = best_path_found[0]
                target_len += 1
            else:
                break
                
        return current_best

    # Iterate through components, largest first
    for comp in components:
        path = find_longest_path_in_comp(comp)
        if len(path) > len(best_path):
            best_path = path
        if time.time() > deadline:
            break
            
    return best_path if best_path else [0]
# EVOLVE-BLOCK-END
