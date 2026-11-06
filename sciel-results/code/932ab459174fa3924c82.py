import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    # Allow 135s for computation to be safe within 150s limit
    deadline = start_time + 135.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []

    for i, spec in enumerate(specs):
        # Allocate remaining time / remaining specs
        time_left = deadline - time.time()
        if time_left <= 0:
            # Fallback for remaining specs: single edges
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        paths = []
        
        # Greedy extraction of long paths with multiple restarts
        while remaining_edges:
            if time.time() > deadline - 2.0:
                # Emergency: convert remaining edges to single paths
                for e in remaining_edges:
                    paths.append(list(e))
                break
            
            best_path = None
            # Get active vertices
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            
            # Try multiple random starting vertices
            vertices_list = list(active_vertices)
            random.shuffle(vertices_list)
            
            # Limit attempts to keep speed up, but try more for better paths
            max_starts = min(20, len(vertices_list))
            
            for start in vertices_list[:max_starts]:
                # Quick check: does start have any remaining edges?
                has_edge = False
                for n in adj[start]:
                    e = (min(start, n), max(start, n))
                    if e in remaining_edges:
                        has_edge = True
                        break
                if not has_edge:
                    continue

                current_path = [start]
                visited = {start}
                current_node = start
                
                while True:
                    # Find valid neighbors
                    valid_neighbors = []
                    for n in adj[current_node]:
                        if n in visited:
                            continue
                        e = (min(current_node, n), max(current_node, n))
                        if e in remaining_edges:
                            valid_neighbors.append(n)
                    
                    if not valid_neighbors:
                        break
                    
                    # Heuristic: choose neighbor with fewest remaining options (Warnsdorff)
                    def count_options(node):
                        count = 0
                        for m in adj[node]:
                            if m in visited:
                                continue
                            e = (min(node, m), max(node, m))
                            if e in remaining_edges:
                                count += 1
                        return count
                    
                    valid_neighbors.sort(key=count_options)
                    next_node = valid_neighbors[0]
                    
                    current_path.append(next_node)
                    visited.add(next_node)
                    current_node = next_node
                
                if len(current_path) > 1:
                    if best_path is None or len(current_path) > len(best_path):
                        best_path = current_path
            
            if best_path:
                paths.append(best_path)
                # Remove edges
                for i in range(len(best_path) - 1):
                    u, v = best_path[i], best_path[i+1]
                    e = (min(u, v), max(u, v))
                    remaining_edges.discard(e)
            else:
                # If no path found (should not happen if remaining_edges is not empty),
                # take a single edge
                if remaining_edges:
                    e = next(iter(remaining_edges))
                    paths.append(list(e))
                    remaining_edges.discard(e)
        
        results.append(paths)

    return results
