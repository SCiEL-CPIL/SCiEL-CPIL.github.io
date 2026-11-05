import time
import sys
from collections import defaultdict

# Increase recursion limit for DFS
sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
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
            
    best_path = []
    start_time = time.time()
    time_limit = 55.0 # 5s buffer
    
    for comp in components:
        if time.time() - start_time > time_limit:
            break
            
        comp_size = len(comp)
        if comp_size <= len(best_path):
            continue
            
        # Check if tree
        edge_count = 0
        for u in comp:
            edge_count += len([v for v in adj[u] if v in comp])
        edge_count //= 2
        
        if edge_count == comp_size - 1:
            # Tree: Find diameter
            # BFS from any node to find farthest, then BFS from farthest
            def bfs_farthest(start, comp_set):
                dist = {start: 0}
                q = [start]
                farthest = start
                max_d = 0
                while q:
                    curr = q.pop(0)
                    d = dist[curr]
                    if d > max_d:
                        max_d = d
                        farthest = curr
                    for nei in adj[curr]:
                        if nei in comp_set and nei not in dist:
                            dist[nei] = d + 1
                            q.append(nei)
                return farthest, max_d, dist
                
            comp_set = set(comp)
            far1, _, _ = bfs_farthest(comp[0], comp_set)
            far2, _, dist2 = bfs_farthest(far1, comp_set)
            
            # Reconstruct path
            # We need the actual path, not just length
            # Let's do a DFS/BFS to get the path between far1 and far2
            # Since it's a tree, path is unique
            def get_path_tree(start, end, comp_set):
                # Parent tracking
                parent = {start: None}
                q = [start]
                while q:
                    curr = q.pop(0)
                    if curr == end:
                        break
                    for nei in adj[curr]:
                        if nei in comp_set and nei not in parent:
                            parent[nei] = curr
                            q.append(nei)
                # Reconstruct
                path = []
                curr = end
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                return path[::-1]
                
            path = get_path_tree(far1, far2, comp_set)
            if len(path) > len(best_path):
                best_path = path
        else:
            # Cyclic component: Use DFS with backtracking
            # We will try to find the longest simple path
            # Heuristic: Start from nodes with degree 1 in the component if possible
            # If no degree 1, start from random
            
            # Filter neighbors to only those in component
            adj_comp = {u: set(v for v in adj[u] if v in comp) for u in comp}
            
            # Identify leaves in the component
            leaves = [u for u in comp if len(adj_comp[u]) == 1]
            
            # Try starting from leaves first, then others
            starts = leaves + [u for u in comp if u not in leaves]
            
            # To avoid redundant work, we can just pick one start?
            # No, the longest path might not start at a leaf.
            # However, for sparse graphs, starting at a leaf is a good heuristic.
            # If we find a path that covers all nodes, we stop.
            
            # We will run DFS from the first leaf (or first node)
            # If the component is small enough, we could try all, but let's stick to one or a few.
            # Given the time limit, we'll run DFS from the best candidate.
            
            # Let's define the DFS
            def dfs_longest_path(start, adj_map, comp_set):
                best_local = [start]
                visited_set = {start}
                
                def dfs(curr, path):
                    nonlocal best_local
                    if len(path) > len(best_local):
                        best_local = path[:]
                        
                    # Pruning: if we can't beat best_local, return
                    # Max possible length is len(comp_set)
                    if len(path) + (len(comp_set) - len(visited_set)) <= len(best_local):
                        return
                        
                    unvisited_neighbors = [v for v in adj_map[curr] if v not in visited_set]
                    
                    if not unvisited_neighbors:
                        return
                        
                    # Forced move optimization
                    if len(unvisited_neighbors) == 1:
                        next_node = unvisited_neighbors[0]
                        visited_set.add(next_node)
                        path.append(next_node)
                        dfs(next_node, path)
                        path.pop()
                        visited_set.remove(next_node)
                    else:
                        # Sort neighbors: prefer those with fewer unvisited neighbors (dead ends first?)
                        # Or just iterate
                        # To improve speed, we can order by degree
                        unvisited_neighbors.sort(key=lambda x: len([y for y in adj_map[x] if y not in visited_set]))
                        for next_node in unvisited_neighbors:
                            visited_set.add(next_node)
                            path.append(next_node)
                            dfs(next_node, path)
                            path.pop()
                            visited_set.remove(next_node)
                            
                dfs(start, [start])
                return best_local

            # Try a few start nodes if time permits
            # Start with leaves
            candidates = starts[:5] # Try top 5
            if not candidates:
                candidates = [comp[0]]
                
            for start_node in candidates:
                if time.time() - start_time > time_limit:
                    break
                local_best = dfs_longest_path(start_node, adj_comp, set(comp))
                if len(local_best) > len(best_path):
                    best_path = local_best
                    # If we found a Hamiltonian path, we can stop
                    if len(best_path) == comp_size:
                        break
                        
    return best_path
