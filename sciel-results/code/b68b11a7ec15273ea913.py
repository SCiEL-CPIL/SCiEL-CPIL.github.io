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

    def greedy_path_cover(edges, mode='min', rng=None):
        """
        Greedy path cover.
        mode: 'min' (prefer low degree), 'max' (prefer high degree), 'random'
        """
        if not edges:
            return []
        
        if rng is None:
            rng = random.Random()
            
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        paths = []
        
        while remaining_edges:
            # Find a start vertex.
            # We should start from a vertex that is likely to lead to a long path.
            # If we use 'min' mode, starting from a leaf is good to ensure we don't get stuck in a cycle?
            # Actually, if we start from a leaf, the path is forced to go in one direction.
            # If we start from a high-degree vertex, we have more choices.
            
            # Let's pick a random start vertex from those with degree > 0
            candidates = [u for u in adj if len(adj[u]) > 0]
            if not candidates:
                break
            
            if mode == 'min':
                # Start from min degree
                start = min(candidates, key=lambda u: len(adj[u]))
            elif mode == 'max':
                # Start from max degree
                start = max(candidates, key=lambda u: len(adj[u]))
            else:
                start = rng.choice(candidates)
                
            path = [start]
            visited_vertices = {start}
            current = start
            
            while True:
                next_v = None
                best_score = -1 if mode == 'max' else float('inf')
                
                neighbors = list(adj[current])
                # Filter to only valid neighbors (edge exists and not visited)
                valid_neighbors = []
                for neighbor in neighbors:
                    edge = (min(current, neighbor), max(current, neighbor))
                    if edge in remaining_edges and neighbor not in visited_vertices:
                        valid_neighbors.append(neighbor)
                        
                if not valid_neighbors:
                    break
                
                if mode == 'random':
                    next_v = rng.choice(valid_neighbors)
                elif mode == 'min':
                    # Prefer lower degree
                    next_v = min(valid_neighbors, key=lambda u: len(adj[u]))
                elif mode == 'max':
                    # Prefer higher degree
                    next_v = max(valid_neighbors, key=lambda u: len(adj[u]))
                    
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
                
            if time.time() > deadline:
                break
                
        return paths

    def solve_graph(edges):
        best_cover = None
        best_count = float('inf')
        
        # Try different modes and random seeds
        modes = ['min', 'max', 'random']
        
        for mode in modes:
            # Run multiple iterations for random and to sample different starts
            for _ in range(50):
                if time.time() > deadline:
                    break
                rng = random.Random()
                cover = greedy_path_cover(edges, mode, rng)
                count = len(cover)
                if count < best_count:
                    best_count = count
                    best_cover = cover
                    
        if best_cover is None:
            # Fallback: trivial cover
            best_cover = [[list(e)] for e in edges]
            
        return best_cover

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[list(e)] for e in edges])
            continue
        edges = instance(*spec)
        cover = solve_graph(edges)
        results.append(cover)
        
    return results
