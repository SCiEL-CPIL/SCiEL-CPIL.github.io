import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    # Fallback: if no edges, return [0]
    if not P['edges']:
        return [0]
        
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
            
    components.sort(key=len, reverse=True)
    
    best_path = [components[0][0]] if components else [0]
    best_len = len(best_path)
    
    start_time = time.time()
    time_limit = 85.0 # Safety margin
    
    # Helper: Greedy path generation with noise
    def greedy_path(start, rng, noise_scale=0.1):
        path = [start]
        visited = {start}
        while True:
            last = path[-1]
            neighbors = [v for v in adj[last] if v not in visited]
            if not neighbors:
                break
            # Score: -len(adj[v] - visited) + noise
            scores = []
            for v in neighbors:
                deg = len(adj[v] - visited)
                score = -deg + rng.gauss(0, noise_scale)
                scores.append((score, v))
            scores.sort(reverse=True)
            next_v = scores[0][1]
            path.append(next_v)
            visited.add(next_v)
        return path

    # Helper: Local Search (2-Opt + Insertion)
    def local_search(path, rng, time_check):
        path = list(path)
        visited = set(path)
        
        while time.time() < time_check:
            improved = False
            
            # 1. Try Insertions
            unvisited = [v for v in range(n) if v not in visited]
            insertion_done = False
            for u in unvisited:
                # Check internal positions
                for i in range(1, len(path)):
                    if (u in adj[path[i-1]]) and (u in adj[path[i]]):
                        path.insert(i, u)
                        visited.add(u)
                        improved = True
                        insertion_done = True
                        break
                if insertion_done:
                    break
                # Check position 0
                if len(path) > 0 and u in adj[path[0]]:
                    path.insert(0, u)
                    visited.add(u)
                    improved = True
                    insertion_done = True
                    break
                # Check position len(path)
                if len(path) > 0 and u in adj[path[-1]]:
                    path.append(u)
                    visited.add(u)
                    improved = True
                    insertion_done = True
                    break
            
            if insertion_done:
                continue
            
            # 2. Try 2-Opt moves
            L = len(path)
            found_2opt = False
            for i in range(0, L):
                for j in range(i+1, L):
                    # Check validity
                    if i > 0:
                        if path[j] not in adj[path[i-1]]:
                            continue
                    else:
                        pass
                    
                    if j < L - 1:
                        if path[i] not in adj[path[j+1]]:
                            continue
                    else:
                        pass
                    
                    # Apply
                    path[i:j+1] = reversed(path[i:j+1])
                    improved = True
                    found_2opt = True
                    break
                if found_2opt:
                    break
            
            if found_2opt:
                continue
                
            # Stuck
            break
            
        return path

    # Phase 1: Greedy starts
    rng = random.Random(42)
    
    # Try all nodes as starts for greedy
    for start in range(n):
        if time.time() > time_limit:
            break
        p = greedy_path(start, rng, noise_scale=0.5)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            
    # Phase 2: Local Search on best candidates
    candidates = []
    for start in range(n):
        if time.time() > time_limit:
            break
        p = greedy_path(start, rng, noise_scale=0.5)
        candidates.append(p)
        
    candidates.sort(key=len, reverse=True)
    top_candidates = candidates[:10]
    
    for p in top_candidates:
        if time.time() > time_limit:
            break
        improved_p = local_search(p, rng, time_limit)
        if len(improved_p) > best_len:
            best_len = len(improved_p)
            best_path = improved_p
            
    # Phase 3: Randomized DFS (if time remains)
    def dfs_limited(start, visited, path, max_depth, time_check):
        nonlocal best_path, best_len
        if time.time() > time_check:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
        if len(path) >= max_depth:
            return
            
        last = path[-1]
        neighbors = list(adj[last])
        neighbors.sort(key=lambda x: len(adj[x]))
        
        for nei in neighbors:
            if nei not in visited:
                visited.add(nei)
                path.append(nei)
                dfs_limited(nei, visited, path, max_depth, time_check)
                path.pop()
                visited.remove(nei)
                
    if components:
        largest_comp = components[0]
        start = largest_comp[0]
        visited = {start}
        path = [start]
        dfs_limited(start, visited, path, 100, time_limit)
        
    return best_path
