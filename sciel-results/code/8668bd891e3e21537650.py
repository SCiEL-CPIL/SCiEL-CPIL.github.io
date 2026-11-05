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
    
    # Find largest connected component
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            comp = []
            stack = [i]
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited_comp[neighbor]:
                        visited_comp[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
    
    components.sort(key=len, reverse=True)
    comp = components[0]
    comp_set = set(comp)
    comp_size = len(comp)
    
    # Build adjacency restricted to component
    adj_comp = [adj[u] & comp_set for u in range(n)]
    
    # Check if tree - use BFS diameter
    edge_count = sum(len(adj_comp[u]) for u in comp) // 2
    if edge_count == comp_size - 1:
        def bfs_diameter(start_node):
            dist = {start_node: 0}
            parent = {start_node: None}
            q = [start_node]
            farthest = start_node
            max_dist = 0
            while q:
                u = q.pop(0)
                if dist[u] > max_dist:
                    max_dist = dist[u]
                    farthest = u
                for v in adj_comp[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
            return farthest, parent
        
        u1, parent1 = bfs_diameter(comp[0])
        u2, parent2 = bfs_diameter(u1)
        path = []
        cur = u2
        while cur is not None:
            path.append(cur)
            cur = parent2[cur]
        path.reverse()
        return path
    
    # Use branch-and-bound with connectivity pruning
    best_path = []
    deadline = time.time() + 45.0
    
    # Precompute degrees for Warnsdorff ordering
    deg = [len(adj_comp[u]) for u in range(n)]
    
    def count_connected(unvisited_mask, current_node):
        """Count connected components in unvisited graph after removing current_node's constraints"""
        # Quick check: are all unvisited nodes reachable from any neighbor of current?
        # For efficiency, just check if the unvisited set is connected
        unvisited_nodes = [i for i in range(n) if unvisited_mask & (1 << i)]
        if not unvisited_nodes:
            return 1
        start = unvisited_nodes[0]
        visited = set()
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            for nb in adj_comp[node]:
                if (unvisited_mask & (1 << nb)) and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        return 1 if len(visited) == len(unvisited_nodes) else 0
    
    # DFS with backtracking, using bitmask for seen
    def dfs(current, path, seen_mask, depth):
        nonlocal best_path, deadline
        
        if time.time() > deadline:
            return
        
        if depth > len(best_path):
            best_path = path[:]
        
        if depth == comp_size:
            return
        
        # Get unvisited neighbors
        unvisited_neighbors = []
        for nb in adj_comp[current]:
            if not (seen_mask & (1 << nb)):
                unvisited_neighbors.append(nb)
        
        if not unvisited_neighbors:
            return
        
        # Warnsdorff ordering: prefer neighbors with fewer unvisited options
        def warnsdorff_key(nb):
            options = 0
            for nn in adj_comp[nb]:
                if not (seen_mask & (1 << nn)):
                    options += 1
            return (options, nb)
        
        unvisited_neighbors.sort(key=warnsdorff_key)
        
        for nb in unvisited_neighbors:
            if time.time() > deadline:
                return
            
            new_seen = seen_mask | (1 << nb)
            new_depth = depth + 1
            
            # Pruning: check if remaining unvisited nodes can still be reached
            remaining = comp_size - new_depth
            if remaining > 0:
                # Check connectivity of unvisited nodes
                unvisited_mask = 0
                for i in range(n):
                    if not (new_seen & (1 << i)) and i in comp_set:
                        unvisited_mask |= (1 << i)
                
                if unvisited_mask:
                    # Quick connectivity check from nb
                    visited_check = set()
                    stack = [nb]
                    visited_check.add(nb)
                    while stack:
                        node = stack.pop()
                        for nn in adj_comp[node]:
                            if (unvisited_mask & (1 << nn)) and nn not in visited_check:
                                visited_check.add(nn)
                                stack.append(nn)
                    
                    if len(visited_check) < remaining + 1:  # +1 because nb is visited but adjacent
                        # Not all remaining nodes reachable, but we might still extend through nb
                        # Only prune if nb itself can't reach any unvisited
                        if len(visited_check) == 1:
                            continue
            
            path.append(nb)
            dfs(nb, path, new_seen, new_depth)
            path.pop()
    
    # Try multiple starting points
    # Start with high-degree nodes first
    start_order = sorted(comp, key=lambda x: -deg[x])
    
    for start in start_order:
        if time.time() > deadline:
            break
        if comp_size <= len(best_path):
            break
        dfs(start, [start], 1 << start, 1)
    
    return best_path if best_path else comp[:1]
# EVOLVE-BLOCK-END
