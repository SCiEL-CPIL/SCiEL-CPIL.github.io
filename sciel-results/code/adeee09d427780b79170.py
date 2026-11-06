import time
import sys
import random

sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    deadline = time.time() + 80  # Safety margin
    
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
    
    degree = [len(adj[i]) for i in range(n)]
    
    def count_reachable(current, visited_set):
        """Count unvisited nodes reachable from current via unvisited nodes only."""
        bstack = [current]
        bseen = {current}
        count = 0
        while bstack:
            u = bstack.pop()
            for v in adj[u]:
                if v in visited_set:
                    continue
                if v not in bseen:
                    bseen.add(v)
                    count += 1
                    bstack.append(v)
        return count
    
    def dfs(path, current, visited, comp_set):
        nonlocal best_path, best_len
        
        if len(path) > best_len:
            best_path = path[:]
            best_len = len(best_path)
        
        # Quick upper bound
        total_unvisited = len(comp_set) - len(visited & comp_set)
        if len(path) + total_unvisited <= best_len:
            return
        
        # Get unvisited neighbors within component
        neighbors = [v for v in adj[current] if v in comp_set and v not in visited]
        
        if not neighbors:
            return
        
        # Prune using reachable count
        reachable_count = count_reachable(current, visited | comp_set - comp_set)  # Just unvisited in comp
        # Actually, let's count reachable unvisited nodes
        unvisited_in_comp = comp_set - visited
        if unvisited_in_comp:
            # Check if all unvisited are reachable from current
            bstack = [current]
            bseen = {current}
            reachable = set()
            while bstack:
                u = bstack.pop()
                for v in adj[u]:
                    if v in unvisited_in_comp and v not in bseen:
                        bseen.add(v)
                        reachable.add(v)
                        bstack.append(v)
            if len(path) + len(reachable) <= best_len:
                return
        
        # Heuristic: Sort by number of unvisited neighbors (Warnsdorff-like)
        def heuristic(v):
            unvis_nei = 0
            for w in adj[v]:
                if w in comp_set and w not in visited:
                    unvis_nei += 1
            return unvis_nei
        
        neighbors.sort(key=heuristic)
        
        for v in neighbors:
            if time.time() >= deadline:
                return
            path.append(v)
            visited.add(v)
            dfs(path, v, visited, comp_set)
            path.pop()
            visited.discard(v)
            if time.time() >= deadline:
                return
    
    # Greedy warm-start to get a good initial path quickly
    def greedy_walk(start, comp_set):
        path = [start]
        visited = {start}
        while True:
            last = path[-1]
            candidates = [v for v in adj[last] if v in comp_set and v not in visited]
            if not candidates:
                break
            # Warnsdorff: pick neighbor with fewest unvisited neighbors
            candidates.sort(key=lambda x: sum(1 for y in adj[x] if y in comp_set and y not in visited))
            next_node = candidates[0]
            path.append(next_node)
            visited.add(next_node)
        return path
    
    # Try greedy from each node in largest component to get good initial bound
    largest_comp = components[0]
    comp_set = set(largest_comp)
    
    for start in sorted(largest_comp, key=lambda x: degree[x]):
        if time.time() >= deadline:
            break
        path = greedy_walk(start, comp_set)
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
    
    # Now do systematic DFS with the improved bound
    # Try starting from nodes with low degree first (leaves are good starting points)
    sorted_starts = sorted(largest_comp, key=lambda x: degree[x])
    
    for start in sorted_starts:
        if time.time() >= deadline:
            break
        path = [start]
        visited = {start}
        dfs(path, start, visited, comp_set)
        if time.time() >= deadline:
            break
    
    return best_path
