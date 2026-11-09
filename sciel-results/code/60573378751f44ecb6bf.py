import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Identify largest connected component to focus search
    visited_comp = [False] * n
    largest_comp = []
    for i in range(n):
        if not visited_comp[i]:
            stack = [i]
            comp = []
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            if len(comp) > len(largest_comp):
                largest_comp = comp
            
    if not largest_comp:
        return [0]
    
    best_path = []
    best_len = 0
    
    # Phase 1: Multi-start greedy with Warnsdorff heuristic
    def greedy_warnsdorff(start_node):
        visited = {start_node}
        path = [start_node]
        while True:
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            # Warnsdorff: pick neighbor with fewest unvisited neighbors
            next_node = min(neighbors, key=lambda x: (len(adj[x] - visited), x))
            path.append(next_node)
            visited.add(next_node)
        return path

    # Try all starting nodes
    for i in largest_comp:
        path = greedy_warnsdorff(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Also try extending paths from both ends
    def extend_path(path, visited):
        """Try to extend a path from both ends."""
        path = list(path)
        visited = set(visited)
        
        # Try extending from the right end
        while True:
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            next_node = min(neighbors, key=lambda x: (len(adj[x] - visited), x))
            path.append(next_node)
            visited.add(next_node)
        
        # Try extending from the left end
        while True:
            first = path[0]
            neighbors = [x for x in adj[first] if x not in visited]
            if not neighbors:
                break
            next_node = min(neighbors, key=lambda x: (len(adj[x] - visited), x))
            path.insert(0, next_node)
            visited.add(next_node)
        
        return path

    # Improve best_path by extending from both ends
    if best_path:
        visited = set(best_path)
        improved = extend_path(best_path, visited)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(best_path)
    
    if time.time() > deadline:
        return best_path

    # Phase 2: Targeted DFS with better pruning
    # Focus on finding paths longer than best_len
    
    def dfs_with_pruning(node, path, visited, target_len, deadline):
        """DFS with dead-end pruning and forced move detection."""
        if time.time() > deadline:
            return None
        if len(path) == target_len:
            return list(path)
        
        # Pruning: if we can't possibly reach target_len, return None
        remaining = target_len - len(path)
        if remaining <= 0:
            return list(path) if len(path) >= target_len else None
        
        # Get unvisited neighbors
        neighbors = [x for x in adj[node] if x not in visited]
        
        if not neighbors:
            return None
        
        # Check for forced moves: neighbors that would become isolated if not visited now
        # A neighbor x is "critical" if it has exactly 1 unvisited neighbor (besides node)
        # Actually, if x has only node as an unvisited neighbor, then x is a dead end
        # If we don't visit x now, we can never visit it later (since node is visited)
        
        critical = []
        for x in neighbors:
            # Count unvisited neighbors of x (excluding node which is already visited)
            unvisited_neighs_of_x = [y for y in adj[x] if y not in visited]
            if len(unvisited_neighs_of_x) == 0:
                # x is a leaf - must visit now or never
                critical.append(x)
            elif len(unvisited_neighs_of_x) == 1:
                # x has one other unvisited neighbor - check if it's critical too
                y = list(unvisited_neighs_of_x)[0]
                # If y also has limited options, x might be critical
                unvisited_neighs_of_y = [z for z in adj[y] if z not in visited]
                if len(unvisited_neighs_of_y) <= 1:
                    critical.append(x)
        
        # If there are critical nodes, try them first
        if critical:
            # Sort critical by remaining degree (Warnsdorff)
            critical.sort(key=lambda x: len(adj[x] - visited))
            for x in critical:
                path.append(x)
                visited.add(x)
                res = dfs_with_pruning(x, path, visited, target_len, deadline)
                if res is not None:
                    return res
                path.pop()
                visited.remove(x)
            # If no critical move worked, try non-critical
            non_critical = [x for x in neighbors if x not in critical]
            non_critical.sort(key=lambda x: len(adj[x] - visited))
            for x in non_critical:
                path.append(x)
                visited.add(x)
                res = dfs_with_pruning(x, path, visited, target_len, deadline)
                if res is not None:
                    return res
                path.pop()
                visited.remove(x)
        else:
            # No critical nodes - use Warnsdorff ordering
            neighbors.sort(key=lambda x: len(adj[x] - visited))
            for x in neighbors:
                path.append(x)
                visited.add(x)
                res = dfs_with_pruning(x, path, visited, target_len, deadline)
                if res is not None:
                    return res
                path.pop()
                visited.remove(x)
        
        return None

    # Try to find progressively longer paths
    current_target = best_len + 1
    max_possible = len(largest_comp)
    
    # Sort start nodes: low-degree nodes are often good endpoints
    start_nodes = sorted(largest_comp, key=lambda x: len(adj[x]))
    
    # Time budget for each target length
    time_per_target = 2.0
    
    while current_target <= max_possible and time.time() < deadline - 5.0:
        search_deadline = min(deadline - 5.0, time.time() + time_per_target)
        
        found_path = None
        
        for start in start_nodes:
            if time.time() > search_deadline:
                break
            
            path = [start]
            visited = {start}
            res = dfs_with_pruning(start, path, visited, current_target, search_deadline)
            if res is not None:
                found_path = res
                break
        
        if found_path:
            best_len = current_target
            best_path = found_path
            current_target += 1
            # Slightly increase time budget for harder targets
            time_per_target = min(time_per_target * 1.2, 5.0)
        else:
            # If we couldn't find a path of length current_target, 
            # it's unlikely we can find longer ones with this approach
            break
    
    # Final improvement: try to extend best_path
    if best_path and time.time() < deadline - 2.0:
        visited = set(best_path)
        improved = extend_path(best_path, visited)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(best_path)
    
    return best_path
# EVOLVE-BLOCK-END
