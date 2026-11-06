import time
from collections import deque

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
    
    # For general components, use enhanced search strategies
    # Strategy 1: Try to find Hamiltonian path using backtracking with heuristics
    
    def try_hamiltonian(start, time_limit):
        """Try to find a path visiting all vertices in the component."""
        path = [start]
        visited_set = {start}
        
        def onward_count(v):
            count = 0
            for w in adj_list[v]:
                if w in comp_set and w not in visited_set:
                    count += 1
            return count
        
        def backtrack(curr, depth):
            if time.time() > time_limit:
                return False
            
            if depth == comp_size:
                nonlocal best_path, best_len
                best_len = comp_size
                best_path = path[:]
                return True
            
            neighbors = [v for v in adj_list[curr] if v in comp_set and v not in visited_set]
            
            if not neighbors:
                return False
            
            # Sort by onward degree (Warnsdorff's rule)
            neighbors.sort(key=lambda x: onward_count(x))
            
            for v in neighbors:
                path.append(v)
                visited_set.add(v)
                
                if backtrack(v, depth + 1):
                    return True
                
                path.pop()
                visited_set.remove(v)
                
                if time.time() > time_limit:
                    return False
            
            return False
        
        return backtrack(start, 1)
    
    # Try Hamiltonian path from low-degree nodes (likely endpoints)
    start_nodes = sorted(largest_comp, key=lambda x: deg[x])
    
    for start in start_nodes[:20]:
        if time.time() > deadline - 30:
            break
        if try_hamiltonian(start, deadline - 30):
            return best_path
    
    # Strategy 2: Longest path using DFS with pruning
    def dfs_longest(curr, path, visited_set, time_limit):
        nonlocal best_path, best_len
        
        if time.time() > time_limit:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        if len(path) == comp_size:
            return
        
        neighbors = [v for v in adj_list[curr] if v in comp_set and v not in visited_set]
        
        if not neighbors:
            return
        
        # Sort by degree (prefer lower degree)
        neighbors.sort(key=lambda x: deg[x])
        
        for v in neighbors:
            path.append(v)
            visited_set.add(v)
            dfs_longest(v, path, visited_set, time_limit)
            path.pop()
            visited_set.remove(v)
            
            if time.time() > time_limit:
                return
    
    # Try DFS from various starting points
    dfs_starts = start_nodes[:15] + sorted(largest_comp, key=lambda x: -deg[x])[:5]
    dfs_starts = list(set(dfs_starts))
    
    for start in dfs_starts:
        if time.time() > deadline - 10:
            break
        
        path = [start]
        visited_set = {start}
        dfs_longest(start, path, visited_set, deadline - 10)
        
        if best_len == comp_size:
            break
    
    # Strategy 3: Greedy extension with local search
    if best_len < comp_size:
        def greedy_extend(path, visited_set, time_limit):
            extended = True
            while extended and time.time() < time_limit:
                extended = False
                
                # Try to extend from the end
                curr = path[-1]
                best_next = None
                best_score = -1
                for v in adj_list[curr]:
                    if v in comp_set and v not in visited_set:
                        onward = sum(1 for w in adj_list[v] if w in comp_set and w not in visited_set and w != curr)
                        score = -onward * 10 + deg[v]
                        if score > best_score:
                            best_score = score
                            best_next = v
                if best_next is not None:
                    path.append(best_next)
                    visited_set.add(best_next)
                    extended = True
                else:
                    # Try to extend from the beginning
                    curr = path[0]
                    best_next = None
                    best_score = -1
                    for v in adj_list[curr]:
                        if v in comp_set and v not in visited_set:
                            onward = sum(1 for w in adj_list[v] if w in comp_set and w not in visited_set and w != curr)
                            score = -onward * 10 + deg[v]
                            if score > best_score:
                                best_score = score
                                best_next = v
                    if best_next is not None:
                        path.insert(0, best_next)
                        visited_set.add(best_next)
                        extended = True
            return path, visited_set
        
        def local_search_insert(path, visited_set, time_limit):
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                unvisited = [v for v in largest_comp if v not in visited_set]
                
                for u in unvisited:
                    if time.time() > time_limit:
                        break
                    
                    # Try to insert u between path[i] and path[i+1]
                    for i in range(len(path) - 1):
                        a = path[i]
                        b = path[i + 1]
                        if u in adj[a] and u in adj[b]:
                            new_path = path[:i+1] + [u] + path[i+1:]
                            new_visited = visited_set | {u}
                            path = new_path
                            visited_set = new_visited
                            improved = True
                            break
                    if improved:
                        break
                
                # Try extending from either end
                if not improved:
                    curr = path[0]
                    for v in adj_list[curr]:
                        if v in comp_set and v not in visited_set:
                            path = [v] + path
                            visited_set = visited_set | {v}
                            improved = True
                            break
                    
                    if not improved:
                        curr = path[-1]
                        for v in adj_list[curr]:
                            if v in comp_set and v not in visited_set:
                                path = path + [v]
                                visited_set = visited_set | {v}
                                improved = True
                                break
            return path, visited_set
        
        # Start from low-degree nodes
        for start in start_nodes[:30]:
            if time.time() > deadline - 5:
                break
            
            path = [start]
            visited_set = {start}
            
            path, visited_set = greedy_extend(path, visited_set, deadline - 10)
            path, visited_set = local_search_insert(path, visited_set, deadline - 5)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            if best_len == comp_size:
                break
        
        # Final polish
        if best_len < comp_size and time.time() < deadline - 1:
            visited_in_path = set(best_path)
            best_path, _ = local_search_insert(best_path, visited_in_path, deadline - 0.5)
            best_len = len(best_path)
    
    if not best_path:
        best_path = [largest_comp[0]]
    
    return best_path
