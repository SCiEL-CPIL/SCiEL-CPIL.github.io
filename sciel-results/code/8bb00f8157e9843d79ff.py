import time
import sys
from collections import deque

# Increase recursion limit
sys.setrecursionlimit(10000)

# P is the pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            stack = [i]
            visited_comp[i] = True
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            components.append(comp)
    
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    best_path = [components[0][0]] if components else [0]
    best_len = len(best_path)
    
    largest_comp = components[0]
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    # Count edges in the largest component
    e_count = 0
    for u in largest_comp:
        for v in adj[u]:
            if v in comp_set and u < v:
                e_count += 1
    
    # If it's a tree, find diameter in O(N)
    if e_count == comp_size - 1:
        def bfs_farthest(start, comp_set):
            dist = {start: 0}
            parent = {start: None}
            queue = deque([start])
            farthest = start
            max_dist = 0
            while queue:
                u = queue.popleft()
                if dist[u] > max_dist:
                    max_dist = dist[u]
                    farthest = u
                for v in adj[u]:
                    if v in comp_set and v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
            path = [farthest]
            while parent[farthest] is not None:
                farthest = parent[farthest]
                path.append(farthest)
            path.reverse()
            return path
        
        path1 = bfs_farthest(largest_comp[0], comp_set)
        end_node_a = path1[-1]
        path2 = bfs_farthest(end_node_a, comp_set)
        
        if len(path2) > best_len:
            best_path = path2
            best_len = len(path2)
        
        return best_path
    
    # For general graphs, use a combination of:
    # 1. Iterative deepening DFS with strong pruning
    # 2. Multiple restarts with different heuristics
    # 3. 2-opt style local search to improve found paths
    
    # Strategy 1: Multiple greedy approaches for good initial paths
    def greedy_warnsdorff(start, visited):
        """Greedy extension using Warnsdorff's rule."""
        path = [start]
        current = start
        while True:
            neighbors = [v for v in adj[current] if v not in visited and v in comp_set]
            if not neighbors:
                break
            def heuristic(v):
                unvis = sum(1 for w in adj[v] if w not in visited and w in comp_set)
                return unvis
            neighbors.sort(key=heuristic)
            current = neighbors[0]
            visited.add(current)
            path.append(current)
        return path
    
    def greedy_randomized(start, visited, seed_offset=0):
        """Greedy with some randomization to explore different paths."""
        import random
        rng = random.Random(seed_offset)
        path = [start]
        current = start
        while True:
            neighbors = [v for v in adj[current] if v not in visited and v in comp_set]
            if not neighbors:
                break
            def heuristic(v):
                unvis = sum(1 for w in adj[v] if w not in visited and w in comp_set)
                return unvis
            neighbors.sort(key=heuristic)
            # Pick from top 3 with some randomness
            top_k = min(3, len(neighbors))
            current = rng.choice(neighbors[:top_k])
            visited.add(current)
            path.append(current)
        return path
    
    # Try multiple greedy strategies from various starts
    for start in largest_comp:
        if time.time() >= deadline:
            break
        visited = {start}
        path = greedy_warnsdorff(start, visited)
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
    
    # Also try randomized greedy from a few starts
    for trial in range(5):
        if time.time() >= deadline:
            break
        import random
        rng = random.Random(trial)
        start = rng.choice(largest_comp)
        visited = {start}
        path = greedy_randomized(start, visited, seed_offset=trial)
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
    
    # Precompute degrees within the component
    deg = {u: sum(1 for v in adj[u] if v in comp_set) for u in largest_comp}
    
    # Strategy 2: DFS with pruning using articulation points
    # Find articulation points in the component
    disc = [-1] * n
    low = [-1] * n
    parent_arr = [-1] * n
    ap = [False] * n
    time_counter = [0]
    
    def tarjan_ap(u):
        if time.time() >= deadline:
            return
        disc[u] = low[u] = time_counter[0]
        time_counter[0] += 1
        children = 0
        for v in adj[u]:
            if v not in comp_set:
                continue
            if disc[v] == -1:
                children += 1
                parent_arr[v] = u
                tarjan_ap(v)
                if time.time() >= deadline:
                    return
                low[u] = min(low[u], low[v])
                if parent_arr[u] == -1 and children > 1:
                    ap[u] = True
                if parent_arr[u] != -1 and low[v] >= disc[u]:
                    ap[u] = True
            elif v != parent_arr[u]:
                low[u] = min(low[u], disc[v])
    
    for u in largest_comp:
        if disc[u] == -1:
            parent_arr[u] = -1
            tarjan_ap(u)
            if time.time() >= deadline:
                break
    
    # DFS with aggressive pruning
    search_done = [False]
    dfs_nodes_visited = [0]
    
    def reachable_count(current, visited):
        """Count unvisited nodes in comp_set reachable from current via unvisited nodes."""
        if current not in comp_set:
            return 0
        stack = [current]
        seen = {current}
        count = 0
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in visited or v not in comp_set:
                    continue
                if v not in seen:
                    seen.add(v)
                    count += 1
                    stack.append(v)
        return count
    
    def dfs_search(path, current, visited, depth):
        nonlocal best_path, best_len, search_done
        if search_done[0] or time.time() >= deadline:
            search_done[0] = True
            return
        
        dfs_nodes_visited[0] += 1
        
        if len(path) > best_len:
            best_path = path[:]
            best_len = len(path)
        
        # If we've found a Hamiltonian path, stop
        if best_len >= comp_size:
            search_done[0] = True
            return
        
        # Quick upper bound
        remaining = comp_size - len(visited)
        if len(path) + remaining <= best_len:
            return
        
        # Get unvisited neighbors
        neighbors = [v for v in adj[current] if v not in visited and v in comp_set]
        if not neighbors:
            return
        
        # Sort by heuristic: prefer nodes with fewer unvisited neighbors (Warnsdorff)
        def heuristic(v):
            unvis = sum(1 for w in adj[v] if w not in visited and w in comp_set)
            return unvis
        
        neighbors.sort(key=heuristic)
        
        for v in neighbors:
            if search_done[0] or time.time() >= deadline:
                search_done[0] = True
                return
            
            # Additional pruning: if v is an articulation point, check connectivity
            # after visiting v. This is expensive, so only do it when close to best_len
            if ap[v] and len(path) + remaining - 1 <= best_len + 2:
                test_visited = visited | {v}
                reach = reachable_count(v, test_visited)
                if len(path) + 1 + reach <= best_len:
                    continue
            
            path.append(v)
            visited.add(v)
            dfs_search(path, v, visited, depth + 1)
            path.pop()
            visited.discard(v)
    
    # Start DFS from nodes sorted by degree
    starts = sorted(largest_comp, key=lambda x: deg[x], reverse=True)
    
    # Adaptive number of starts based on time
    num_starts = min(len(starts), 30)
    
    for i, start in enumerate(starts[:num_starts]):
        if time.time() >= deadline:
            break
        if search_done[0]:
            break
        path = [start]
        visited = {start}
        dfs_search(path, start, visited, 1)
        
        # If time is running low, stop
        if time.time() > deadline - 15:
            break
    
    # Strategy 3: Try to extend the best path found using 2-opt style local search
    if best_len < comp_size and time.time() < deadline - 10:
        # Try to extend from both ends of best_path
        best_set = set(best_path)
        
        # Try extending from the end
        for node in [best_path[-1], best_path[0]]:
            if time.time() >= deadline - 5:
                break
            extended = False
            for neighbor in adj[node]:
                if neighbor not in best_set and neighbor in comp_set:
                    # Try to insert this neighbor and continue
                    if node == best_path[-1]:
                        new_path = best_path + [neighbor]
                        new_set = best_set | {neighbor}
                        # Try to extend further
                        current = neighbor
                        while True:
                            if time.time() >= deadline - 5:
                                break
                            candidates = [v for v in adj[current] if v not in new_set and v in comp_set]
                            if not candidates:
                                break
                            # Pick the one with fewest options (Warnsdorff)
                            def h(v):
                                return sum(1 for w in adj[v] if w not in new_set and w in comp_set)
                            candidates.sort(key=h)
                            current = candidates[0]
                            new_set.add(current)
                            new_path.append(current)
                            extended = True
                        if len(new_path) > best_len:
                            best_path = new_path
                            best_len = len(new_path)
                    elif node == best_path[0]:
                        new_path = [neighbor] + best_path
                        new_set = best_set | {neighbor}
                        current = neighbor
                        while True:
                            if time.time() >= deadline - 5:
                                break
                            candidates = [v for v in adj[current] if v not in new_set and v in comp_set]
                            if not candidates:
                                break
                            def h(v):
                                return sum(1 for w in adj[v] if w not in new_set and w in comp_set)
                            candidates.sort(key=h)
                            current = candidates[0]
                            new_set.add(current)
                            new_path.insert(0, current)
                            extended = True
                        if len(new_path) > best_len:
                            best_path = new_path
                            best_len = len(new_path)
                    break  # Only try one extension per end
    
    return best_path

# EVOLVE-BLOCK-END
