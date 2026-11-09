"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(n, p, seed):
        edges = instance(n, p, seed)
        m = len(edges)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Degrees
        deg = {v: len(adj[v]) for v in range(n)}
        
        # Odd degree vertices
        odd_verts = [v for v in range(n) if deg[v] % 2 == 1]
        
        # Greedy path cover
        # Strategy: Start from odd vertices, then even vertices with high degree
        # Use a greedy approach to extend paths
        
        used_edges = set()
        paths = []
        
        # Sort starting vertices: odd first, then by degree descending
        start_order = sorted(odd_verts, key=lambda v: -deg[v]) + \
                      sorted([v for v in range(n) if deg[v] % 2 == 0], key=lambda v: -deg[v])
        
        for start in start_order:
            if time.time() > deadline:
                break
            
            # Find an unused edge incident to start
            start_edge = None
            for v in adj[start]:
                e = tuple(sorted((start, v)))
                if e not in used_edges:
                    start_edge = (start, v)
                    break
            
            if start_edge is None:
                continue
            
            # Extend path greedily
            path = [start, start_edge[1]]
            used_edges.add(tuple(sorted(start_edge)))
            
            current = start_edge[1]
            prev = start
            
            while True:
                # Try to extend from current
                extended = False
                for v in sorted(adj[current], key=lambda x: -deg[x]):
                    if v == prev:
                        continue
                    e = tuple(sorted((current, v)))
                    if e not in used_edges:
                        path.append(v)
                        used_edges.add(e)
                        prev = current
                        current = v
                        extended = True
                        break
                
                if not extended:
                    break
            
            paths.append(path)
        
        # Check if all edges are covered
        if len(used_edges) < m:
            # Cover remaining edges as single-edge paths
            for e in edges:
                if e not in used_edges:
                    paths.append(list(e))
                    used_edges.add(e)
        
        return paths

    # Solve each graph
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: return single-edge paths for remaining graphs
            for remaining in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results):]:
                n, p, seed = remaining
                edges = instance(n, p, seed)
                results.append([list(e) for e in edges])
            break
        results.append(solve_graph(*spec))
    
    return results
# EVOLVE-BLOCK-END
