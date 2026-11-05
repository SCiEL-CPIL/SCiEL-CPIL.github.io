import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
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
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
            
    # Sort components by size, largest first
    components.sort(key=len, reverse=True)
    
    best_path = []
    deadline = time.time() + 50.0 # Safety margin for 60s limit
    
    # We primarily care about the largest component
    # But let's check all, just in case (though unlikely to be better than largest)
    
    for comp in components:
        if time.time() > deadline:
            break
            
        # If the component is small, we can try exhaustive or near-exhaustive search
        # If it's large, we use heuristic-guided DFS
        
        # For this specific graph size (N=100), the largest component is likely < 50 nodes
        # A simple DFS with backtracking might be fast enough.
        
        # Let's use a DFS that tries to extend the path.
        # We'll start from each node in the component? No, that's too slow.
        # We'll use a "randomized restart" or "multiple starts" strategy.
        
        # Actually, a better approach for sparse graphs:
        # The longest path in a tree is the diameter.
        # If the component is a tree, we can find it in O(N).
        # Let's check if the component is a tree.
        
        comp_set = set(comp)
        m_comp = 0
        for u in comp:
            m_comp += len(adj[u] & comp_set)
        m_comp //= 2
        
        if m_comp == len(comp) - 1:
            # It's a tree. Find diameter.
            # Diameter is the longest path.
            # Pick an arbitrary node, find farthest node A.
            # From A, find farthest node B.
            # Path from A to B is the diameter.
            
            def bfs_farthest(start, comp_set, adj):
                dist = {start: 0}
                parent = {start: None}
                queue = [start]
                farthest = start
                max_dist = 0
                while queue:
                    u = queue.pop(0)
                    if dist[u] > max_dist:
                        max_dist = dist[u]
                        farthest = u
                    for v in adj[u]:
                        if v in comp_set and v not in dist:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                # Reconstruct path
                path = []
                curr = farthest
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                return path[::-1]
            
            # Find diameter
            start_node = comp[0]
            path1 = bfs_farthest(start_node, comp_set, adj)
            end_node = path1[-1]
            path2 = bfs_farthest(end_node, comp_set, adj)
            
            if len(path2) > len(best_path):
                best_path = path2
                
        else:
            # Not a tree. Use DFS with backtracking.
            # Since it's sparse, we can try a limited DFS.
            
            # To improve performance, we can use a heuristic to order neighbors.
            # We want to find a long path.
            
            # Let's try starting from nodes with high degree first?
            # Or just run a DFS from each node in the component?
            # With N~30-50, this might be feasible if we prune.
            
            # Pruning: if current_path_len + (len(comp) - len(current_path)) <= len(best_path), prune.
            
            # We'll run a DFS from each node in the component, but with a time check.
            
            # To make it faster, we can use a "branch and bound" where we keep the global best.
            
            def dfs(node, path, visited_in_dfs):
                nonlocal best_path
                nonlocal deadline
                
                if time.time() > deadline:
                    return
                
                if len(path) > len(best_path):
                    best_path = path[:]
                    
                # Upper bound: current length + remaining nodes in component
                remaining = len(comp) - len(visited_in_dfs)
                if len(path) + remaining <= len(best_path):
                    return
                    
                for neighbor in adj[node]:
                    if neighbor in comp_set and neighbor not in visited_in_dfs:
                        visited_in_dfs.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited_in_dfs)
                        path.pop()
                        visited_in_dfs.remove(neighbor)
                        
            # Try starting from each node in the component
            # To speed up, we can limit the number of starts or use a heuristic.
            # Let's just iterate through all nodes in the component.
            for start in comp:
                if time.time() > deadline:
                    break
                # Heuristic: sort neighbors by degree?
                # We'll just run DFS.
                visited_in_dfs = {start}
                path = [start]
                dfs(start, path, visited_in_dfs)
                
    return best_path
# EVOLVE-BLOCK-END
