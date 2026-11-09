import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
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
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not visited_comp[v]:
                        visited_comp[v] = True
                        stack.append(v)
            components.append(comp)
    
    best_path = []
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    # Function to compute longest path in a component using DFS
    def find_longest_path_in_comp(comp):
        nonlocal best_path, deadline
        if not comp:
            return
        
        # If component is small, we can be more exhaustive or just use DFS
        # Build local adjacency
        local_adj = {u: set() for u in comp}
        for u in comp:
            for v in adj[u]:
                if v in local_adj:
                    local_adj[u].add(v)
        
        # If it's a tree, calculate diameter
        # Check if tree: edges == nodes - 1
        edge_count = 0
        for u in comp:
            edge_count += len(local_adj[u])
        edge_count //= 2
        if edge_count == len(comp) - 1:
            # It's a tree. Find diameter.
            # Pick arbitrary node
            start_node = comp[0]
            # BFS to find farthest
            def bfs_farthest(start):
                dist = {start: 0}
                parent = {start: -1}
                queue = [start]
                farthest = start
                max_dist = 0
                while queue:
                    u = queue.pop(0)
                    if dist[u] > max_dist:
                        max_dist = dist[u]
                        farthest = u
                    for v in local_adj[u]:
                        if v not in dist:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                return farthest, dist, parent
            
            # 1. Find A
            A, dist1, parent1 = bfs_farthest(start_node)
            # 2. Find B from A
            B, dist2, parent2 = bfs_farthest(A)
            
            # Reconstruct path
            path = [B]
            cur = B
            while cur != A:
                cur = parent2[cur]
                path.append(cur)
            path.reverse() # A to B
            if len(path) > len(best_path):
                best_path = path
            return

        # If not a tree (has cycles), use DFS with pruning
        # We try starting from each node in the component
        # To save time, we can limit the number of starts or use heuristics
        # Let's try all nodes if time permits, otherwise a subset
        
        def dfs(u, visited, path):
            nonlocal best_path, deadline
            if time.time() > deadline:
                return
            if len(path) > len(best_path):
                best_path = path[:]
            
            neighbors = [v for v in local_adj[u] if v not in visited]
            if not neighbors:
                return
            
            # Heuristic: sort by degree in unvisited graph (Warnsdorff)
            # Lower degree first
            def get_unvisited_degree(v):
                return sum(1 for w in local_adj[v] if w not in visited)
            
            neighbors.sort(key=get_unvisited_degree)
            
            for v in neighbors:
                visited.add(v)
                path.append(v)
                dfs(v, visited, path)
                path.pop()
                visited.remove(v)
                if time.time() > deadline:
                    return

        # Try starting from each node
        # Optimization: Start from leaves (degree 1) first as they are likely endpoints
        starts = sorted(comp, key=lambda x: len(local_adj[x]))
        
        for start in starts:
            if time.time() > deadline:
                break
            visited = {start}
            path = [start]
            dfs(start, visited, path)

    for comp in components:
        find_longest_path_in_comp(comp)
        if time.time() > deadline:
            break

    if not best_path:
        # Fallback: return a single vertex or simple edge
        if P['edges']:
            best_path = [P['edges'][0][0], P['edges'][0][1]]
        else:
            best_path = [0]
            
    return best_path
# EVOLVE-BLOCK-END
