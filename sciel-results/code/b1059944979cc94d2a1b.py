    def decompose(edges):
        if not edges:
            return []
        
        remaining_edges = set(edges)
        pieces = []
        
        def find_longest_fundamental_cycle(edge_set):
            if not edge_set:
                return None
            
            # Build adjacency
            adj = defaultdict(list)
            for u, v in edge_set:
                adj[u].append(v)
                adj[v].append(u)
                
            # Find a spanning tree (e.g., using BFS)
            vertices = list(adj.keys())
            if not vertices:
                return None
            root = vertices[0]
            
            visited = {root}
            parent = {root: -1}
            queue = deque([root])
            tree_edges = set()
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        parent[v] = u
                        tree_edges.add((min(u,v), max(u,v)))
                        queue.append(v)
                        
            # Identify non-tree edges
            non_tree_edges = [e for e in edge_set if e not in tree_edges]
            
            best_cycle = None
            best_len = 0
            
            for e in non_tree_edges:
                u, v = e
                # Find path from u to v in the tree
                # We can do this by climbing parents
                path_u = []
                curr = u
                while curr != -1:
                    path_u.append(curr)
                    curr = parent[curr]
                    
                path_v = []
                curr = v
                while curr != -1:
                    path_v.append(curr)
                    curr = parent[curr]
                    
                # Find LCA
                set_u = set(path_u)
                lca = None
                for node in path_v:
                    if node in set_u:
                        lca = node
                        break
                        
                if lca is None:
                    continue # Should not happen if connected
                    
                # Reconstruct path
                # Path from u to lca
                path_1 = []
                curr = u
                while curr != lca:
                    path_1.append(curr)
                    curr = parent[curr]
                path_1.append(lca)
                
                # Path from v to lca
                path_2 = []
                curr = v
                while curr != lca:
                    path_2.append(curr)
                    curr = parent[curr]
                path_2.append(lca)
                
                # Cycle is path_1 + path_2 reversed (excluding duplicate lca)
                cycle = path_1 + path_2[-2::-1]
                
                if len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                    
            return best_cycle

        while True:
            cycle = find_longest_fundamental_cycle(remaining_edges)
            if not cycle:
                break
                
            # Remove edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                remaining_edges.discard((min(u,v), max(u,v)))
                
            pieces.append(cycle)
            
        for u, v in remaining_edges:
            pieces.append([u, v])
            
        return pieces
