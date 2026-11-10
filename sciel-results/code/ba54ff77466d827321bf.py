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

    def get_odd_vertices(adj):
        return [u for u in adj if len(adj[u]) % 2 != 0]

    def greedy_path_cover_odd_guided(edges, rng):
        """
        Greedy path cover that prioritizes starting and ending at odd-degree vertices.
        This aims to achieve the theoretical minimum number of paths (trails).
        """
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        paths = []
        
        while remaining_edges:
            # Determine start vertex
            odd_verts = get_odd_vertices(adj)
            if odd_verts:
                # Pick a random odd vertex
                start = rng.choice(odd_verts)
            else:
                # All degrees even, pick any vertex with edges
                candidates = [u for u in adj if adj[u]]
                if not candidates:
                    break
                start = rng.choice(candidates)
                
            path = [start]
            visited_vertices = {start}
            current = start
            
            while True:
                valid_neighbors = []
                for neighbor in adj[current]:
                    edge = (min(current, neighbor), max(current, neighbor))
                    if edge in remaining_edges and neighbor not in visited_vertices:
                        valid_neighbors.append(neighbor)
                        
                if not valid_neighbors:
                    break
                
                # Choose next vertex
                # Prefer neighbors that are odd-degree (in the current remaining graph)
                # This helps close the path at an odd vertex, which is desirable for the decomposition
                odd_neighbors = [v for v in valid_neighbors if len(adj[v]) % 2 != 0]
                
                if odd_neighbors:
                    # Among odd neighbors, maybe pick one with highest degree? Or random?
                    # Random is fine to explore
                    next_v = rng.choice(odd_neighbors)
                else:
                    # If no odd neighbors, pick one with max degree to keep options open?
                    # Or just random
                    next_v = rng.choice(valid_neighbors)
                    
                path.append(next_v)
                visited_vertices.add(next_v)
                current = next_v
                
            if len(path) > 1:
                paths.append(path)
                
            # Remove edges used in this path
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
        
        # Run guided greedy with different random seeds
        for seed in range(50):
            if time.time() > deadline:
                break
            rng = random.Random(seed)
            cover = greedy_path_cover_odd_guided(edges, rng)
            
            count = len(cover)
            if count < best_count:
                best_count = count
                best_cover = cover
        
        if best_cover is None:
            # Fallback: trivial cover (each edge is a path)
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
