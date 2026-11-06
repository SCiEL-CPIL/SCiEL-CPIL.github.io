import time
from collections import deque
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    adj_list = [list(s) for s in adj]
    
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
                for nei in adj_list[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
    
    largest_comp = max(components, key=len)
    comp_size = len(largest_comp)
    comp_set = set(largest_comp)
    
    if comp_size == 0:
        return [0]
    if comp_size == 1:
        return largest_comp[:]
    
    deg = {u: len([v for v in adj_list[u] if v in comp_set]) for u in largest_comp}
    e_count = sum(deg[u] for u in largest_comp) // 2
    is_tree = (e_count == comp_size - 1)
    
    best_path = []
    best_len = 0
    
    deadline = time.time() + 85.0
    
    if is_tree:
        def bfs_farthest(start):
            dist = {start: 0}
            parent = {start: None}
            q = deque([start])
            farthest = start
            max_d = 0
            while q:
                u = q.popleft()
                if dist[u] > max_d:
                    max_d = dist[u]
                    farthest = u
                for v in adj_list[u]:
                    if v in comp_set and v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
            return farthest, parent
        
        u, _ = bfs_farthest(largest_comp[0])
        v, parent = bfs_farthest(u)
        
        path = []
        curr = v
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        best_path = path
        best_len = len(path)
        return best_path
    
    # Iterative DFS with Warnsdorff heuristic
    def iterative_dfs(start, time_limit):
        nonlocal best_path, best_len
        
        # Stack-based DFS
        # Each stack entry: (node, path_so_far, visited_set, neighbor_index)
        # We use a more memory-efficient approach
        
        path = [start]
        visited_set = {start}
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        if len(path) == comp_size:
            return True
        
        # Use a stack of (node, path, visited, next_neighbor_idx)
        # But to save memory, we'll use a different approach:
        # Keep a single path and backtrack
        
        # Actually, let's use a proper iterative DFS with explicit stack
        # Stack entries: (current_node, path_copy, visited_copy, next_nei_idx)
        # This is expensive in memory. Let's use a different strategy.
        
        # Strategy: Use a single path array and a stack of "decisions"
        # Each decision records which neighbor was chosen at each level
        
        # For efficiency, let's limit the depth of backtracking
        # and use a greedy-with-limited-backtracking approach
        
        max_backtrack_depth = 30
        
        # We'll use a recursive-like iterative approach
        # with a stack that stores (node, path_len, visited_set_snapshot, next_nei_idx)
        
        # Actually, for a 100-node graph, let's just use recursion with sys.setrecursionlimit
        # but with a time check
        
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(5000)
        
        def dfs_recursive(curr, path, visited_set):
            nonlocal best_path, best_len
            
            if time.time() > time_limit:
                return False
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            if len(path) == comp_size:
                return True
            
            # Pruning: if even visiting all remaining can't beat best
            remaining = comp_size - len(visited_set)
            if len(path) + remaining <= best_len:
                return False
            
            neighbors = [v for v in adj_list[curr] if v in comp_set and v not in visited_set]
            
            if not neighbors:
                return False
            
            # Warnsdorff: sort by fewest onward moves
            def onward_count(v):
                count = 0
                for w in adj_list[v]:
                    if w in comp_set and w not in visited_set and w != curr:
                        count += 1
                return count
            
            neighbors.sort(key=lambda x: (onward_count(x), deg[x]))
            
            for v in neighbors:
                path.append(v)
                visited_set.add(v)
                
                if dfs_recursive(v, path, visited_set):
                    path.pop()
                    visited_set.remove(v)
                    return True
                
                path.pop()
                visited_set.remove(v)
                
                if time.time() > time_limit:
                    return False
            
            return False
        
        try:
            result = dfs_recursive(start, path, visited_set)
            return result
        finally:
            sys.setrecursionlimit(old_limit)
    
    # Try multiple starting nodes
    start_nodes = sorted(largest_comp, key=lambda x: deg[x])
    
    # First try low-degree nodes (more likely to be endpoints)
    for start in start_nodes[:20]:
        if time.time() > deadline:
            break
        if iterative_dfs(start, deadline):
            if best_len == comp_size:
                break
    
    # If not Hamiltonian, try to improve with local search
    if best_len < comp_size and time.time() < deadline:
        # Try to insert unvisited vertices into the path
        # For each unvisited vertex, try to insert it between consecutive path vertices
        # if both edges exist
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            visited_in_path = set(best_path)
            unvisited = [v for v in largest_comp if v not in visited_in_path]
            
            for u in unvisited:
                if time.time() > deadline:
                    break
                
                # Try to insert u between path[i] and path[i+1]
                for i in range(len(best_path) - 1):
                    a = best_path[i]
                    b = best_path[i + 1]
                    if u in adj[a] and u in adj[b]:
                        # Insert u between a and b
                        new_path = best_path[:i+1] + [u] + best_path[i+1:]
                        if len(new_path) > best_len:
                            best_path = new_path
                            best_len = len(new_path)
                            improved = True
                            break
                if improved:
                    break
            
            # Also try to extend from either end
            if not improved:
                # Try extending from start
                curr = best_path[0]
                for v in adj_list[curr]:
                    if v in comp_set and v not in set(best_path):
                        new_path = [v] + best_path
                        if len(new_path) > best_len:
                            best_path = new_path
                            best_len = len(new_path)
                            improved = True
                            break
                
                # Try extending from end
                if not improved:
                    curr = best_path[-1]
                    for v in adj_list[curr]:
                        if v in comp_set and v not in set(best_path):
                            new_path = best_path + [v]
                            if len(new_path) > best_len:
                                best_path = new_path
                                best_len = len(new_path)
                                improved = True
                                break
    
    # Final attempt: try more starting nodes if we have time
    if best_len < comp_size and time.time() < deadline - 5:
        high_degree_starts = sorted(largest_comp, key=lambda x: -deg[x])
        for start in high_degree_starts[:10]:
            if time.time() > deadline - 2:
                break
            if start in set(best_path):
                continue
            iterative_dfs(start, deadline - 1)
            if best_len == comp_size:
                break
    
    if not best_path:
        best_path = [largest_comp[0]]
    
    return best_path
