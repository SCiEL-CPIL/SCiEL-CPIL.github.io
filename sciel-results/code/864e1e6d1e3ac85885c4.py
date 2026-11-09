def construct():
    import random
    import time
    import sys

    def instance_graph(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_instance(n, p, seed):
        edges = instance_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency and degree
        adj = {i: set() for i in range(n)}
        deg = {i: 0 for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            edge_set.add((u, v))
        
        remaining_edges = set(edge_set)
        paths = []
        
        # To ensure we finish within time, we just run the greedy loop
        # The graph is small, so this is fast.
        
        while remaining_edges:
            # Find a start vertex
            # Prefer vertices with highest remaining degree
            # We need to calculate remaining degrees
            rem_deg = {i: 0 for i in range(n)}
            for u, v in remaining_edges:
                rem_deg[u] += 1
                rem_deg[v] += 1
            
            start = None
            max_deg = -1
            for i in range(n):
                if rem_deg[i] > max_deg:
                    max_deg = rem_deg[i]
                    start = i
            
            if start is None or max_deg == 0:
                break # Should not happen if remaining_edges is not empty
            
            # Greedy walk
            path = [start]
            used_vertices = {start}
            current = start
            
            while True:
                # Find available neighbors
                candidates = []
                for neighbor in adj[current]:
                    if (min(current, neighbor), max(current, neighbor)) in remaining_edges:
                        if neighbor not in used_vertices:
                            # Heuristic: prefer neighbor with higher remaining degree?
                            # Or lower? 
                            # Higher degree neighbor might have more options later.
                            candidates.append(neighbor)
                
                if not candidates:
                    break
                
                # Choose next
                # Sort by remaining degree descending
                candidates.sort(key=lambda x: rem_deg[x], reverse=True)
                next_v = candidates[0]
                
                # Add to path
                path.append(next_v)
                used_vertices.add(next_v)
                
                # Remove edge
                edge_tuple = (min(current, next_v), max(current, next_v))
                remaining_edges.remove(edge_tuple)
                
                # Update rem_deg
                rem_deg[current] -= 1
                rem_deg[next_v] -= 1
                
                current = next_v
                
            paths.append(path)
            
        return paths

    res1 = solve_instance(20, 0.3, 1)
    res2 = solve_instance(25, 0.3, 2)
    res3 = solve_instance(30, 0.25, 3)
    
    return [res1, res2, res3]
