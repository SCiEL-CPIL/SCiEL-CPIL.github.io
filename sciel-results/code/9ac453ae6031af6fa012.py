import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def build_adj(edge_set):
        adj = {}
        for u, v in edge_set:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj

    def find_cycle_heuristic(adj, start, rng, max_attempts=10, max_depth=30):
        """
        Attempts to find a long cycle starting from 'start' using randomized greedy walks.
        Returns the longest cycle found as a list of vertices, or None.
        """
        best_cycle = None
        
        for _ in range(max_attempts):
            path = [start]
            on_path = {start}
            
            for _ in range(max_depth):
                current = path[-1]
                neighbors = [n for n in adj.get(current, []) if n not in on_path]
                
                if not neighbors:
                    break
                
                # Bias towards higher degree neighbors to keep graph connected
                # But allow some randomness to escape local minima
                # Score: degree * random_factor
                scored_neighbors = []
                for n in neighbors:
                    deg = len(adj.get(n, []))
                    # Use a soft max or just degree with noise
                    score = deg * (0.8 + 0.4 * rng.random())
                    scored_neighbors.append((score, n))
                
                scored_neighbors.sort(key=lambda x: -x[0])
                next_node = scored_neighbors[0][1]
                
                path.append(next_node)
                on_path.add(next_node)
                
                # Check if we can close the cycle
                if len(path) >= 3 and start in adj.get(next_node, []):
                    if best_cycle is None or len(path) > len(best_cycle):
                        best_cycle = list(path)
                    break
            
            # If we didn't close in the loop, check if the final node connects to start
            if len(path) >= 3 and start in adj.get(path[-1], []):
                if best_cycle is None or len(path) > len(best_cycle):
                    best_cycle = list(path)
        
        return best_cycle

    def find_any_cycle(adj, start=None):
        """
        Finds any cycle in the graph using DFS.
        If start is provided, tries to find a cycle involving it or reachable from it.
        """
        visited = set()
        
        def dfs(node, parent, path):
            visited.add(node)
            for neighbor in adj.get(node, []):
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    # Found a back edge to a visited node
                    # The cycle is from the index of neighbor in path to end
                    idx = path.index(neighbor)
                    return path[idx:]
                path.append(neighbor)
                result = dfs(neighbor, node, path)
                if result:
                    return result
                path.pop()
            return None
        
        # If start is given, start DFS from there
        if start is not None and start in adj:
            result = dfs(start, -1, [start])
            if result:
                return result
        
        # Otherwise, search the whole graph
        for v in sorted(adj.keys()):
            if v not in visited and adj[v]:
                result = dfs(v, -1, [v])
                if result:
                    return result
        return None

    def decompose(edges):
        if not edges:
            return []
        
        pieces = []
        remaining = set()
        for e in edges:
            remaining.add(tuple(sorted(e)))
        
        rng = Random(42) # Fixed seed for reproducibility within decompose
        
        while remaining:
            if time.time() - start_time > DEADLINE:
                break
            
            adj = build_adj(remaining)
            
            # Find vertices with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                break
            
            # Pick the vertex with the highest degree
            start_vertex = max(candidates, key=lambda v: len(adj[v]))
            
            # Try heuristic cycle finding first
            cycle = find_cycle_heuristic(adj, start_vertex, rng)
            
            if cycle is None:
                # Fallback to finding any cycle
                cycle = find_any_cycle(adj, start_vertex)
            
            if cycle is None:
                # No cycle found, graph is a forest (or components are trees)
                # We must break and handle remaining edges as single pieces
                break
            
            # Validate and remove cycle edges
            cycle_edges = set()
            valid = True
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i+1) % len(cycle)]
                key = tuple(sorted((u, v)))
                if key not in remaining:
                    valid = False
                    break
                cycle_edges.add(key)
            
            if valid:
                pieces.append(cycle)
                remaining -= cycle_edges
            else:
                # If validation fails (shouldn't happen if adj is correct), break to avoid infinite loop
                break
        
        # Add remaining edges as single pieces
        for e in remaining:
            pieces.append([e[0], e[1]])
        
        return pieces
    
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() - start_time > DEADLINE:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
