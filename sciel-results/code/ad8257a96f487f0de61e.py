import time
import sys

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

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
    
    # Sort components by size
    components.sort(key=len, reverse=True)
    
    if not components:
        return [0]
        
    # We only care about the largest component for the longest path
    # Any path must be contained within a single connected component.
    largest_comp = components[0]
    if len(largest_comp) == 1:
        return largest_comp
        
    # Map vertices to indices in the largest component for efficiency?
    # Or just work with the global indices but restrict search to this component.
    
    # Pre-compute neighbors sorted by degree (descending) for heuristic ordering
    # Higher degree nodes are often better to visit later (to keep options open)
    # Or earlier? Warnsdorff suggests visiting neighbors with *fewest* unvisited neighbors first.
    # Let's stick with dynamic ordering based on unvisited neighbors.
    
    adj_list = [list(adj[i]) for i in range(n)]
    
    # Helper to get unvisited neighbors of a node
    def get_unvisited_neighbors(node, visited_set):
        return [v for v in adj_list[node] if v not in visited_set and v in largest_comp_set]

    largest_comp_set = set(largest_comp)
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_path = []
    best_len = 0
    
    # Global variables for DFS
    # Using a list for path and a set for visited is standard.
    # For pruning, we need to check connectivity of the unvisited subgraph.
    
    def count_components(subgraph_nodes):
        """Count connected components in a set of nodes."""
        if not subgraph_nodes:
            return 0
        visited_sub = set()
        comp_count = 0
        for start in subgraph_nodes:
            if start not in visited_sub:
                comp_count += 1
                # BFS/DFS
                stack = [start]
                visited_sub.add(start)
                while stack:
                    node = stack.pop()
                    for nei in adj_list[node]:
                        if nei in subgraph_nodes and nei not in visited_sub:
                            visited_sub.add(nei)
                            stack.append(nei)
        return comp_count

    # Pruning heuristic:
    # If the unvisited subgraph has more than 2 connected components, we cannot visit all of them
    # in a single path extension from the current ends.
    # However, we don't need to visit *all* unvisited nodes, just enough to beat best_len.
    # A stronger bound: 
    # Let U be the set of unvisited nodes.
    # Let C1, C2, ... be the components of U.
    # The current path has two ends: path[0] and path[-1].
    # We can extend from path[0] into a component containing a neighbor of path[0].
    # We can extend from path[-1] into a component containing a neighbor of path[-1].
    # If a component is not reachable from either end, it is "dead" and we can ignore it.
    # The maximum additional vertices we can visit is the sum of the sizes of the components
    # that are reachable from at least one end.
    # If current_len + max_additional < best_len, we can prune.
    
    def max_additional_vertices(path, visited_set):
        """
        Estimate max vertices we can still add.
        Returns 0 if no extension possible.
        """
        if not path:
            return 0
            
        end1 = path[0]
        end2 = path[-1]
        
        # Get unvisited nodes
        unvisited = [v for v in largest_comp if v not in visited_set]
        if not unvisited:
            return 0
            
        # Find components of unvisited subgraph
        visited_sub = set()
        components = []
        for start in unvisited:
            if start not in visited_sub:
                comp = []
                stack = [start]
                visited_sub.add(start)
                while stack:
                    node = stack.pop()
                    comp.append(node)
                    for nei in adj_list[node]:
                        if nei in unvisited_set and nei not in visited_sub:
                            visited_sub.add(nei)
                            stack.append(nei)
                components.append(comp)
        
        # For each component, check if it's reachable from end1 or end2
        reachable_size = 0
        for comp in components:
            comp_set = set(comp)
            # Check if end1 has a neighbor in this component
            reach1 = any(nei in comp_set for nei in adj_list[end1] if nei in visited_set or nei in comp_set)
            # Actually, just check if end1 is adjacent to any node in comp
            reach1 = False
            for nei in adj_list[end1]:
                if nei in comp_set:
                    reach1 = True
                    break
            reach2 = False
            for nei in adj_list[end2]:
                if nei in comp_set:
                    reach2 = True
                    break
            
            if reach1 or reach2:
                reachable_size += len(comp)
                
        return reachable_size

    # To make max_additional_vertices faster, we can optimize the component finding.
    # But N=100 is small.
    
    # We need unvisited_set for fast lookup in max_additional
    unvisited_set = set() # Will be maintained in DFS
    
    def dfs(path, visited_set, unvisited_set):
        nonlocal best_path, best_len, deadline
        
        if time.time() > deadline:
            return
            
        current_len = len(path)
        
        # Update best
        if current_len > best_len:
            best_len = current_len
            best_path = list(path)
            
        # Pruning: Check if we can beat best_len
        # If current_len + max_additional <= best_len, prune
        # Note: max_additional calculation is O(N^2) worst case, which is fine for N=100.
        # But we do it at every node. 100^2 * 100^depth? No, depth is <= 100.
        # 100 * 100 * 100 = 1e6 operations per node? No, component finding is O(N+E).
        # It might be slow if we do it at every step.
        # Let's only do this check if current_len + (n - current_len) > best_len (trivial)
        # Actually, if current_len + (number of unvisited) <= best_len, prune. This is trivial.
        # The component check is more powerful.
        
        # Optimization: Only run the expensive component check if current_len + len(unvisited_set) > best_len
        if current_len + len(unvisited_set) <= best_len:
            return
            
        # Heuristic ordering of neighbors
        last = path[-1]
        # Also check start of path? Usually we extend from the end.
        # We can also extend from the start.
        # Let's try extending from the last node first.
        
        neighbors = [v for v in adj_list[last] if v in unvisited_set]
        
        if not neighbors:
            # Try extending from the start?
            # This is a "double-ended" growth.
            # But standard DFS usually just extends one end.
            # If we can't extend the end, we might be stuck.
            # We can try to extend from path[0] as well.
            start_node = path[0]
            neighbors_start = [v for v in adj_list[start_node] if v in unvisited_set]
            if neighbors_start:
                 # Sort neighbors_start
                 def heuristic(v):
                     # Count unvisited neighbors of v
                     return len([u for u in adj_list[v] if u in unvisited_set])
                 neighbors_start.sort(key=heuristic)
                 for v in neighbors_start:
                     if time.time() > deadline:
                         return
                     new_path = [v] + path
                     new_visited = visited_set | {v}
                     new_unvisited = unvisited_set - {v}
                     dfs(new_path, new_visited, new_unvisited)
            return
            
        # Sort neighbors by heuristic: fewer unvisited neighbors first (Warnsdorff)
        def heuristic(v):
            return len([u for u in adj_list[v] if u in unvisited_set])
        
        neighbors.sort(key=heuristic)
        
        for v in neighbors:
            if time.time() > deadline:
                return
                
            # Pruning before recursing
            # Estimate max gain from this branch
            # If we go to v, the new unvisited set is unvisited_set - {v}
            # We can approximate the max additional gain as len(unvisited_set) - 1
            # But the component structure matters.
            # For now, let's just recurse. The check at the top of dfs will prune.
            
            new_path = path + [v]
            new_visited = visited_set | {v}
            new_unvisited = unvisited_set - {v}
            
            dfs(new_path, new_visited, new_unvisited)
            
    # Start DFS from each vertex in the largest component
    # Sort starts by degree (descending) to find long paths early
    starts = sorted(largest_comp, key=lambda x: len(adj_list[x]), reverse=True)
    
    for start in starts:
        if time.time() > deadline:
            break
            
        path = [start]
        visited_set = {start}
        unvisited_set = largest_comp_set - visited_set
        
        dfs(path, visited_set, unvisited_set)
        
        # Early exit if we found a Hamiltonian path in the largest component
        if best_len == len(largest_comp):
            break

    # Fallback: if no path found (should not happen), return a single vertex
    if not best_path:
        best_path = [largest_comp[0]]
        
    return best_path
