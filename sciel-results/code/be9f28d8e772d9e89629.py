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
    
    # Build adjacency lists for iteration
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
    
    # Get the largest component
    largest_comp = max(components, key=len)
    comp_size = len(largest_comp)
    comp_set = set(largest_comp)
    
    if comp_size == 0:
        return [0]
    if comp_size == 1:
        return largest_comp[:]
    
    # Compute degrees within the component
    deg = {u: len([v for v in adj_list[u] if v in comp_set]) for u in largest_comp}
    
    # Check if the component is a tree
    e_count = sum(deg[u] for u in largest_comp) // 2
    is_tree = (e_count == comp_size - 1)
    
    best_path = []
    
    if is_tree:
        # Find diameter of the tree using BFS
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
    else:
        # Graph has cycles. Use a combination of strategies.
        
        # Strategy 1: Greedy with backtracking (DFS with pruning)
        # Strategy 2: Try multiple starts with Warnsdorff-like heuristic
        # Strategy 3: If greedy finds a very long path, try to extend it
        
        start_nodes = sorted(largest_comp, key=lambda x: deg[x])
        
        best_len = 0
        deadline = time.time() + 85.0
        
        def dfs_path(start, path, visited_set):
            nonlocal best_path, best_len
            
            if time.time() > deadline:
                return False
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            if len(path) == comp_size:
                return True
            
            curr = path[-1]
            remaining = comp_size - len(visited_set)
            
            if len(path) + remaining <= best_len:
                return False
            
            # Get unvisited neighbors
            neighbors = [v for v in adj_list[curr] if v in comp_set and v not in visited_set]
            
            if not neighbors:
                return False
            
            # Warnsdorff heuristic: try neighbors with fewest onward moves first
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
                
                if dfs_path(v, path, visited_set):
                    path.pop()
                    visited_set.remove(v)
                    return True
                
                path.pop()
                visited_set.remove(v)
                
                if time.time() > deadline:
                    return False
            
            return False
        
        # Try DFS from several promising starting nodes
        for start in start_nodes[:15]:
            if time.time() > deadline:
                break
            if dfs_path(start, [start], {start}):
                if best_len == comp_size:
                    break
            # Even if DFS didn't find Hamiltonian, it may have updated best_path
        
        # If we found a Hamiltonian path, we're done
        if best_len == comp_size:
            return best_path
        
        # Strategy: Try to extend the best path found so far from both ends
        if len(best_path) > 0:
            # Try extending from the start of the path
            extended_start = best_path[:]
            visited_set = set(extended_start)
            while True:
                curr = extended_start[0]
                nexts = [v for v in adj_list[curr] if v in comp_set and v not in visited_set]
                if not nexts:
                    break
                # Pick the neighbor with fewest onward moves
                def onward_count(v):
                    count = 0
                    for w in adj_list[v]:
                        if w in comp_set and w not in visited_set:
                            count += 1
                    return count
                nexts.sort(key=lambda x: (onward_count(x), deg[x]))
                extended_start.insert(0, nexts[0])
                visited_set.add(nexts[0])
            
            # Try extending from the end of the path
            extended_end = best_path[:]
            visited_set = set(extended_end)
            while True:
                curr = extended_end[-1]
                nexts = [v for v in adj_list[curr] if v in comp_set and v not in visited_set]
                if not nexts:
                    break
                def onward_count(v):
                    count = 0
                    for w in adj_list[v]:
                        if w in comp_set and w not in visited_set:
                            count += 1
                    return count
                nexts.sort(key=lambda x: (onward_count(x), deg[x]))
                extended_end.append(nexts[0])
                visited_set.add(nexts[0])
            
            # Combine: try both extensions
            if len(extended_start) > best_len:
                best_path = extended_start
                best_len = len(extended_start)
            if len(extended_end) > best_len:
                best_path = extended_end
                best_len = len(extended_end)
            
            # Try combining: extend from one end of best_path, then the other
            # This is tricky, so we'll just use the best single extension
        
        # If still not good enough, try a different approach: 
        # Start from a high-degree node and do a more aggressive search
        if best_len < comp_size * 0.8 and time.time() < deadline:
            # Try starting from higher-degree nodes
            high_degree_starts = sorted(largest_comp, key=lambda x: -deg[x])
            for start in high_degree_starts[:10]:
                if time.time() > deadline:
                    break
                if start in set(best_path):
                    continue
                if dfs_path(start, [start], {start}):
                    if best_len == comp_size:
                        break
                if time.time() > deadline:
                    break
        
        if not best_path:
            # Fallback: just return a single node
            best_path = [largest_comp[0]]
    
    return best_path
