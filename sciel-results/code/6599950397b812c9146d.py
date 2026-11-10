import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def min_path_cover_greedy(edges):
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        paths = []
        
        while remaining_edges:
            # Find a start vertex. Prefer degree 1.
            start = None
            min_deg = float('inf')
            # Iterate over a copy of keys to avoid mutation issues if we were removing nodes, but we aren't
            for u in list(adj.keys()):
                deg = len(adj[u])
                if deg > 0:
                    if deg < min_deg:
                        min_deg = deg
                        start = u
                    if deg == 1:
                        break
            if start is None:
                break
                
            path = [start]
            visited_vertices = {start}
            current = start
            
            while True:
                next_v = None
                min_next_deg = float('inf')
                # Sort neighbors for determinism? Not strictly needed but good for reproducibility
                for neighbor in sorted(adj[current]):
                    edge = (min(current, neighbor), max(current, neighbor))
                    if edge in remaining_edges:
                        if neighbor not in visited_vertices:
                            deg = len(adj[neighbor])
                            if deg < min_next_deg:
                                min_next_deg = deg
                                next_v = neighbor
                            if deg == 1:
                                break
                if next_v is None:
                    break
                path.append(next_v)
                visited_vertices.add(next_v)
                current = next_v
                
            if len(path) > 1:
                paths.append(path)
                
            # Remove edges
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = (min(u, v), max(u, v))
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
                    adj[u].remove(v)
                    adj[v].remove(u)
                else:
                    # Should not happen
                    pass
                
            if time.time() > deadline:
                break
                
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[list(e)] for e in edges])
            continue
        edges = instance(*spec)
        cover = min_path_cover_greedy(edges)
        # Fallback if greedy failed completely (shouldn't happen)
        if not cover:
            cover = [[list(e)] for e in edges]
        results.append(cover)
        
    return results
