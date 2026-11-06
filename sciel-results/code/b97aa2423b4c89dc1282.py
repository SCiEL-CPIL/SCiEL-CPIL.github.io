import time
import sys
from collections import defaultdict
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        """
        Decompose graph into edge-disjoint simple paths covering all edges.
        Strategy:
        1. Use a multi-start greedy approach with randomized tie-breaking.
        2. Use dynamic edge weights to guide path extension.
           - Prefer edges leading to vertices with higher remaining degree.
           - Penalize edges that lead to "dead ends" (vertices that would become 
             isolated or have very low connectivity in the remaining graph).
        3. Keep the best result found across multiple restarts.
        """
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        original_edges = set(tuple(sorted(e)) for e in edges)
        
        def get_best_decomposition():
            best_paths = None
            best_count = float('inf')
            
            # Number of restarts depends on available time
            # For small graphs, we can afford more restarts
            num_restarts = 50
            
            for _ in range(num_restarts):
                if time.time() > deadline:
                    break
                    
                remaining_edges = set(original_edges)
                paths = []
                
                # Track remaining degree for each vertex
                rem_deg = defaultdict(int)
                for u, v in remaining_edges:
                    rem_deg[u] += 1
                    rem_deg[v] += 1
                
                while remaining_edges:
                    # Pick a starting vertex
                    # Heuristic: Pick vertex with highest remaining degree
                    # Add some randomness to escape local optima
                    candidates = [v for v in rem_deg if rem_deg[v] > 0]
                    if not candidates:
                        break
                    
                    # Sort by degree, but add random tie-breaking
                    # To increase diversity in restarts
                    max_deg = max(rem_deg[v] for v in candidates)
                    top_candidates = [v for v in candidates if rem_deg[v] == max_deg]
                    start = random.choice(top_candidates)
                    
                    # Build the path
                    path = [start]
                    used_in_path = {start}
                    current = start
                    path_edges = []
                    
                    while True:
                        # Find neighbors of current that have an unused edge to current
                        candidates = []
                        for nb in adj[current]:
                            edge = tuple(sorted((current, nb)))
                            if edge in remaining_edges and nb not in used_in_path:
                                # Score the candidate
                                # Prefer higher remaining degree
                                # Also consider if this move leads to a dead end
                                # A simple heuristic: score = rem_deg[nb]
                                # More sophisticated: if rem_deg[nb] == 1, it's a leaf, 
                                # which is good for ending a path, but bad if we are 
                                # forced to end early.
                                score = rem_deg[nb]
                                candidates.append((score, random.random(), nb))
                        
                        if not candidates:
                            break
                        
                        # Pick the best candidate
                        # Sort by score descending, then random value for tie-breaking
                        candidates.sort(key=lambda x: (-x[0], -x[1]))
                        _, _, next_v = candidates[0]
                        
                        path.append(next_v)
                        used_in_path.add(next_v)
                        edge = tuple(sorted((current, next_v)))
                        path_edges.append(edge)
                        
                        # Remove edge from remaining
                        remaining_edges.discard(edge)
                        rem_deg[current] -= 1
                        rem_deg[next_v] -= 1
                        
                        current = next_v
                    
                    # The path is complete.
                    paths.append(path)
                    
                # Update best if this decomposition is better
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = [p[:] for p in paths]
                
                # Early stopping if we hit the theoretical lower bound
                # Lower bound is roughly max(1, odd_vertices/2) but for simple paths 
                # it's constrained by graph structure. 
                # For now, just check if we improved.
                
            return best_paths if best_paths else []

        return get_best_decomposition()

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline:
            break
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
    
    # If we ran out of time, fill remaining with trivial edge paths
    while len(results) < 3:
        spec = specs[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results
