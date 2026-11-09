import time
from random import Random
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110  # 10s safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_simple_path(edges, n):
        """
        Finds a simple path covering as many edges as possible.
        Returns the path (list of vertices) and the set of edges used.
        """
        if not edges:
            return [], set()
        
        # Build adjacency for current edges
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find start vertex: prefer leaves (degree 1), then min degree
        # If no leaves, pick min degree
        min_deg = min(len(adj[v]) for v in adj if adj[v])
        candidates = [v for v in adj if len(adj[v]) == min_deg]
        
        best_path = []
        best_edges = set()
        
        # Try starting from each candidate (usually few)
        # To save time, we can just pick one, or try all if few
        # For n=30, trying all min-degree vertices is fine.
        
        for start in candidates:
            path = [start]
            visited = {start}
            current = start
            used_edges = set()
            
            while True:
                # Get neighbors in current graph
                # We need to know which edges are still available.
                # Since we are constructing a simple path, we just can't revisit vertices.
                # The edges are implicitly defined by the graph structure.
                # We remove edges as we go? No, we just track visited vertices.
                # But we need to ensure we don't use an edge twice?
                # In a simple path, vertices are distinct, so edges are distinct automatically (unless multi-edges, which we don't have).
                
                next_candidates = [nb for nb in adj[current] if nb not in visited]
                if not next_candidates:
                    break
                
                # Heuristic: choose neighbor with minimum degree in the *remaining* graph?
                # Or minimum degree in the original graph?
                # Let's use degree in the original graph as a static heuristic, 
                # or dynamic: degree in the subgraph induced by unvisited vertices?
                # Dynamic is better.
                
                # Calculate dynamic degree for neighbors
                # This is O(n) per step, which is fine for n=30.
                best_next = None
                min_dyn_deg = float('inf')
                
                for nb in next_candidates:
                    # Count unvisited neighbors of nb
                    dyn_deg = sum(1 for x in adj[nb] if x not in visited)
                    if dyn_deg < min_dyn_deg:
                        min_dyn_deg = dyn_deg
                        best_next = nb
                
                if best_next is None:
                    break
                
                # Add edge
                u, v = sorted((current, best_next))
                used_edges.add((u, v))
                
                path.append(best_next)
                visited.add(best_next)
                current = best_next
            
            if len(path) > len(best_path):
                best_path = path
                best_edges = used_edges
                
        return best_path, best_edges

    def path_cover(edges, n):
        if not edges:
            return []
        
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            # Find longest simple path in remaining graph
            # We need to pass the subgraph induced by remaining_edges
            # My find_longest_simple_path uses the full 'adj' built from 'edges'.
            # I need to pass the remaining edges to build the correct adjacency.
            
            path, used_edges = find_longest_simple_path(list(remaining_edges), n)
            
            if not path:
                # Should not happen if remaining_edges is not empty
                # Fallback: pick any edge
                e = next(iter(remaining_edges))
                paths.append(list(e))
                remaining_edges.discard(e)
                continue
            
            paths.append(path)
            remaining_edges.difference_update(used_edges)
            
            # Check for timeout
            if time.time() > DEADLINE:
                # Add remaining edges as single-edge paths
                for e in remaining_edges:
                    paths.append(list(e))
                break
        
        return paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        if time.time() > DEADLINE:
            # Emergency fallback
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        n = spec[0]
        
        # Run path cover
        paths = path_cover(edges, n)
        results.append(paths)
        
    return results
