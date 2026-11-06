import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        """
        Decompose graph into edge-disjoint simple paths covering all edges.
        Strategy: Iteratively find the longest simple path in the remaining graph
        using DFS with pruning, remove its edges, repeat.
        For sparse ER graphs with n=20-30, this is feasible and yields far fewer
        paths than greedy.
        """
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        while remaining_edges:
            if time.time() > deadline:
                break
            
            # Find all vertices with at least one remaining edge
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if not active_vertices:
                break
            
            # Try to find the longest simple path in the remaining graph
            # Use DFS with backtracking, but limit search for efficiency
            best_path = None
            best_len = 0
            
            # For each starting vertex, try to extend to find a long path
            # We'll use a greedy DFS that tries to go as far as possible
            # but with some backtracking to find longer paths
            
            # Sort active vertices by degree (descending) for better pruning
            deg = {v: len([e for e in remaining_edges if v in e]) for v in active_vertices}
            start_candidates = sorted(active_vertices, key=lambda v: deg.get(v, 0), reverse=True)
            
            # Try from each start candidate
            found_good = False
            for start in start_candidates:
                if time.time() > deadline:
                    break
                
                # DFS to find longest simple path starting from 'start'
                # Use iterative DFS with state tracking
                best_from_start = [start]
                best_len_from_start = 1
                
                # State: (current_vertex, path_so_far, set_of_used_vertices)
                # We'll use recursion with a depth limit to avoid exponential blowup
                def dfs(current, path, used_verts, depth):
                    nonlocal best_from_start, best_len_from_start, found_good
                    
                    if len(path) > best_len_from_start:
                        best_len_from_start = len(path)
                        best_from_start = list(path)
                        if best_len_from_start >= 5:  # Good enough heuristic
                            found_good = True
                    
                    if depth > 20:  # Limit depth to avoid explosion
                        return
                    
                    # Try all neighbors with remaining edges
                    for nb in adj[current]:
                        edge = tuple(sorted((current, nb)))
                        if edge in remaining_edges and nb not in used_verts:
                            path.append(nb)
                            used_verts.add(nb)
                            dfs(nb, path, used_verts, depth + 1)
                            path.pop()
                            used_verts.remove(nb)
                            if found_good and best_len_from_start >= 10:
                                return  # Early exit if we found a long path
                
                dfs(start, [start], {start}, 0)
                
                if best_from_start and len(best_from_start) > best_len:
                    best_path = best_from_start
                    best_len = best_len_from_start
                
                # If we found a reasonably long path, stop searching other starts
                if best_len >= 8:
                    break
            
            if best_path is None or len(best_path) < 2:
                # Fallback: take any single edge
                e = next(iter(remaining_edges))
                best_path = [e[0], e[1]]
            elif len(best_path) == 1:
                # Shouldn't happen if remaining_edges is non-empty, but handle it
                e = next(iter(remaining_edges))
                best_path = [e[0], e[1]]
            
            # Remove edges of the best path from remaining_edges
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i + 1])))
                remaining_edges.discard(e)
            
            paths.append(best_path)
        
        # If we ran out of time, fill remaining edges as single-edge paths
        if remaining_edges:
            for u, v in remaining_edges:
                paths.append([u, v])
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
    
    # If we ran out of time, fill remaining with trivial edge paths
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results
