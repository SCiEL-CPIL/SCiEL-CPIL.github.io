import sys
import time
import random
import collections

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
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
            
    # Sort components by size, largest first
    components.sort(key=len, reverse=True)
    
    best_path = []
    deadline = time.time() + 80.0  # Safety margin
    
    # Function to find longest path in a component using DFS
    def find_longest_path_in_component(comp):
        nonlocal best_path
        if not comp:
            return
        
        comp_set = set(comp)
        # Heuristic: Start from nodes with low degree? Or high degree?
        # Let's try all nodes in the component as starting points, but limit work per start.
        
        # Optimization: If the component is a tree, compute diameter
        # Count edges in component
        edge_count = 0
        for u in comp:
            edge_count += len(adj[u] & comp_set)
        edge_count //= 2
        
        if edge_count == len(comp) - 1:
            # It's a tree. Find diameter.
            # BFS from arbitrary node to find farthest node u
            def bfs_farthest(start):
                dist = {start: 0}
                q = collections.deque([start])
                farthest = start
                while q:
                    node = q.popleft()
                    for nei in adj[node]:
                        if nei in comp_set and nei not in dist:
                            dist[nei] = dist[node] + 1
                            q.append(nei)
                            if dist[nei] > dist[farthest]:
                                farthest = nei
                return farthest, dist
                
            u, _ = bfs_farthest(comp[0])
            v, dist_v = bfs_farthest(u)
            # Reconstruct path from u to v
            # We need parent pointers
            def bfs_path(start, target):
                prev = {start: None}
                q = collections.deque([start])
                while q:
                    node = q.popleft()
                    if node == target:
                        break
                    for nei in adj[node]:
                        if nei in comp_set and nei not in prev:
                            prev[nei] = node
                            q.append(nei)
                path = []
                curr = target
                while curr is not None:
                    path.append(curr)
                    curr = prev[curr]
                return path[::-1]
            
            path = bfs_path(u, v)
            if len(path) > len(best_path):
                best_path = path
            return

        # If not a tree, use DFS with backtracking
        # To avoid recursion depth issues and overhead, we use an iterative approach or set limit
        # Given N=100, recursion is fine.
        
        # We will run DFS from each node in the component, but stop if time is up
        # We use a heuristic ordering: neighbors with more unvisited neighbors are preferred
        
        # To make it faster, we'll just do a few randomized DFS runs
        # Or a single deep DFS with good ordering
        
        # Let's try a DFS that explores all possibilities but prunes if current_len + remaining_nodes <= best_len
        # This is weak pruning.
        
        # Better: Just do a greedy-like DFS but with backtracking on the last few steps?
        # No, let's just do a standard DFS with time limit.
        
        # We'll pick a start node. The seed program tries all starts.
        # We'll try a subset of starts or randomize.
        
        # Let's try to find the longest path using a simple DFS from every node,
        # but we'll limit the depth or time per start.
        
        # Actually, for sparse graphs, the longest path is often close to the diameter of the "tree-like" core.
        # Let's try BFS from every node to find the longest path in the sense of "longest simple path" is NP-hard.
        # But for N=100, we can do better.
        
        # Let's use a heuristic: 
        # 1. Remove leaves iteratively to find the "core"? No, that finds the 2-core.
        # 2. Just run DFS.
        
        # We'll run DFS from each node in the component.
        # To speed up, we'll sort neighbors by degree (descending) to find long paths quickly.
        
        def dfs(node, visited_mask, path):
            nonlocal best_path
            if time.time() > deadline:
                return
            if len(path) > len(best_path):
                best_path = path[:]
            
            # Pruning: if current length + number of unvisited nodes in comp <= best_len, return
            # This is expensive to compute exactly, so we skip it or use a weak bound
            # Weak bound: len(path) + (len(comp) - len(visited_mask)) <= len(best_path)
            if len(path) + (len(comp) - len(visited_mask)) <= len(best_path):
                return
                
            # Get unvisited neighbors
            unvisited_neis = [nei for nei in adj[node] if nei in comp_set and nei not in visited_mask]
            
            # Sort by degree (descending) to prioritize "open" nodes
            unvisited_neis.sort(key=lambda x: len(adj[x]), reverse=True)
            
            for nei in unvisited_neis:
                if time.time() > deadline:
                    return
                visited_mask.add(nei)
                path.append(nei)
                dfs(nei, visited_mask, path)
                path.pop()
                visited_mask.remove(nei)
                
        # Run DFS from each node in the component
        for start in comp:
            if time.time() > deadline:
                break
            visited_mask = {start}
            path = [start]
            dfs(start, visited_mask, path)
            
    # Process components
    for comp in components:
        if time.time() > deadline:
            break
        find_longest_path_in_component(comp)
        
    # If best_path is empty (shouldn't happen), return any edge
    if not best_path:
        if edges:
            return [edges[0][0], edges[0][1]]
        else:
            return [0]
            
    return best_path
# EVOLVE-BLOCK-END
