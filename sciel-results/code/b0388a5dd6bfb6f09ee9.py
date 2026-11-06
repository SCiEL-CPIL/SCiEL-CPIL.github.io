    def decompose_graph(edges):
        """Decompose edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []
        
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        pieces = []
        
        while remaining_edges:
            # Find a cycle in the remaining graph
            cycle = None
            
            # Get vertices
            vertices = set()
            for u, v in remaining_edges:
                vertices.add(u)
                vertices.add(v)
                
            # Try to find a cycle starting from each unvisited vertex
            visited_global = set()
            
            for start in vertices:
                if start in visited_global:
                    continue
                    
                # DFS from start
                visited = {start}
                parent = {start: -1}
                stack = [start]
                
                while stack and not cycle:
                    node = stack.pop()
                    # Find neighbors in remaining_edges
                    # Optimization: Pre-compute adjacency for remaining_edges?
                    # For small E, iterating is fine.
                    neighbors = []
                    for u, v in remaining_edges:
                        if u == node:
                            neighbors.append(v)
                        elif v == node:
                            neighbors.append(u)
                            
                    for nb in neighbors:
                        if nb == parent[node]:
                            continue
                        if nb in visited:
                            # Found a cycle
                            cycle_nodes = [node]
                            curr = nb
                            while curr != node:
                                cycle_nodes.append(curr)
                                curr = parent[curr]
                            cycle = cycle_nodes
                            break
                        else:
                            visited.add(nb)
                            parent[nb] = node
                            stack.append(nb)
                            
                visited_global.update(visited)
                if cycle:
                    break
            
            if cycle:
                # Extract edges from cycle
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = (min(u, v), max(u, v))
                    remaining_edges.discard(edge)
                pieces.append(cycle)
            else:
                # No cycle found in any component, remaining graph is a forest
                for e in remaining_edges:
                    pieces.append([e[0], e[1]])
                remaining_edges = set()
                break
                
        return pieces
