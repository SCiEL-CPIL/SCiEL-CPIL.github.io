"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    # Deadline guard: 150s wall-clock limit, keep safety margin
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0  # seconds before deadline to stop
    
    def time_left():
        return DEADLINE - (time.time() - start_time) - SAFETY_MARGIN
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def find_longest_path_greedy(adj, start, used_edges):
        """Find a long simple path starting from 'start' using greedy with Warnsdorff's rule."""
        path = [start]
        visited = {start}
        current = start
        
        while True:
            # Get unvisited neighbors
            neighbors = [v for v in adj[current] if (min(current, v), max(current, v)) not in used_edges and v not in visited]
            if not neighbors:
                break
            
            # Warnsdorff's rule: choose neighbor with fewest onward options
            def onward_count(v):
                count = 0
                for w in adj[v]:
                    if w not in visited and (min(v, w), max(v, w)) not in used_edges:
                        count += 1
                return count
            
            neighbors.sort(key=lambda v: onward_count(v))
            next_v = neighbors[0]
            
            edge = (min(current, next_v), max(current, next_v))
            used_edges.add(edge)
            path.append(next_v)
            visited.add(next_v)
            current = next_v
        
        return path
    
    def decompose_graph_matching(edges, max_time=40.0):
        """
        Decompose edges into minimum number of paths using matching-based approach.
        We model this as: find a set of directed paths that cover all edges.
        Each edge can be traversed in one of two directions.
        We want to maximize the number of "chained" edges (where end of one path = start of next).
        
        Approach: Use greedy with local search / simulated annealing to find good path decomposition.
        Start with a random edge ordering, then try to merge paths where possible.
        """
        if not edges:
            return []
        
        adj = build_adjacency(edges)
        total_edges = len(edges)
        vertices = sorted(adj.keys())
        
        # Build edge list with both directions
        directed_edges = []
        for u, v in edges:
            directed_edges.append((u, v))
            directed_edges.append((v, u))
        
        # Greedy approach: repeatedly find longest unused path
        # But use a smarter strategy: prioritize extending existing paths
        
        best_paths = None
        best_count = total_edges
        
        # Multiple restarts with different random seeds
        for seed in range(20):
            if time_left() < 3.0:
                break
            
            rng = random.Random(seed)
            used_edges = set()
            paths = []
            
            # Start from vertices in random order, but prefer low-degree vertices
            # Low-degree vertices are more likely to be endpoints of paths
            vertex_order = sorted(vertices, key=lambda v: (len(adj[v]), rng.random()))
            
            for start in vertex_order:
                if time_left() < 3.0:
                    break
                if len(used_edges) >= total_edges:
                    break
                
                # Try to find a path starting from 'start'
                path = find_longest_path_greedy(adj, start, used_edges)
                if len(path) > 1:
                    paths.append(path)
            
            # Check coverage
            if len(used_edges) == total_edges:
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = [list(p) for p in paths]
            
            # Try to improve by merging paths
            if best_paths is not None and len(paths) > best_count:
                # Try local search: swap endpoints or reorder
                improved = True
                while improved and time_left() > 3.0:
                    improved = False
                    for i in range(len(paths)):
                        if time_left() < 3.0:
                            break
                        for j in range(i + 1, len(paths)):
                            # Try to merge paths[i] and paths[j]
                            p1, p2 = paths[i], paths[j]
                            # Check if end of p1 = start of p2
                            if p1[-1] == p2[0]:
                                merged = p1 + p2[1:]
                                if len(set(merged)) == len(merged):
                                    # Check all edges are valid and not used elsewhere
                                    valid = True
                                    for k in range(len(merged) - 1):
                                        e = (min(merged[k], merged[k+1]), max(merged[k], merged[k+1]))
                                        if e not in used_edges:
                                            valid = False
                                            break
                                    if valid:
                                        paths[i] = merged
                                        del paths[j]
                                        improved = True
                                        break
                            # Check if end of p2 = start of p1
                            if p2[-1] == p1[0]:
                                merged = p2 + p1[1:]
                                if len(set(merged)) == len(merged):
                                    valid = True
                                    for k in range(len(merged) - 1):
                                        e = (min(merged[k], merged[k+1]), max(merged[k], merged[k+1]))
                                        if e not in used_edges:
                                            valid = False
                                            break
                                    if valid:
                                        paths[i] = merged
                                        del paths[j]
                                        improved = True
                                        break
                            # Check if start of p1 = end of p2
                            if p1[0] == p2[-1]:
                                merged = p2[::-1] + p1[1:]
                                if len(set(merged)) == len(merged):
                                    valid = True
                                    for k in range(len(merged) - 1):
                                        e = (min(merged[k], merged[k+1]), max(merged[k], merged[k+1]))
                                        if e not in used_edges:
                                            valid = False
                                            break
                                    if valid:
                                        paths[i] = merged
                                        del paths[j]
                                        improved = True
                                        break
                            # Check if start of p2 = end of p1
                            if p2[0] == p1[-1]:
                                merged = p1 + p2[1:]
                                if len(set(merged)) == len(merged):
                                    valid = True
                                    for k in range(len(merged) - 1):
                                        e = (min(merged[k], merged[k+1]), max(merged[k], merged[k+1]))
                                        if e not in used_edges:
                                            valid = False
                                            break
                                    if valid:
                                        paths[i] = merged
                                        del paths[j]
                                        improved = True
                                        break
        
        # Fallback: simple edge-by-edge decomposition
        if best_paths is None or len(best_paths) >= best_count:
            used_edges = set()
            paths = []
            
            for e in edges:
                if time_left() < 1.0:
                    break
                if e in used_edges:
                    continue
                
                u, v = e
                extended = False
                
                for i, path in enumerate(paths):
                    # Try to prepend u
                    if path[0] == v:
                        new_path = [u] + path
                        if len(set(new_path)) == len(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                    # Try to append u
                    if path[-1] == v:
                        new_path = path + [u]
                        if len(set(new_path)) == len(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                    # Try to prepend v
                    if path[0] == u:
                        new_path = [v] + path
                        if len(set(new_path)) == len(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                    # Try to append v
                    if path[-1] == u:
                        new_path = path + [v]
                        if len(set(new_path)) == len(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                
                if not extended:
                    paths.append([u, v])
                    used_edges.add(e)
            
            if len(used_edges) == total_edges:
                if best_paths is None or len(paths) < best_count:
                    best_paths = [list(p) for p in paths]
        
        return best_paths if best_paths is not None else [[list(e)] for e in edges]
    
    def decompose_graph(edges, max_time=10.0):
        """Decompose edges into minimum number of paths using greedy with restarts."""
        if not edges:
            return []
        
        adj = build_adjacency(edges)
        total_edges = len(edges)
        
        best_paths = None
        best_count = total_edges  # Upper bound: each edge as its own path
        
        # Try multiple random orderings of starting vertices
        vertices = list(adj.keys())
        rng = random.Random(42)
        
        # First, try deterministic ordering
        for order in [vertices, sorted(vertices), list(reversed(vertices))]:
            if time_left() < 2.0:
                break
                
            used_edges = set()
            paths = []
            
            for start in order:
                if time_left() < 2.0:
                    break
                if not used_edges or len(used_edges) < total_edges:
                    # Find an unused edge incident to start
                    incident_unused = [(u, v) for u, v in edges 
                                      if u == start or v == start
                                      and (u, v) not in used_edges]
                    if not incident_unused:
                        continue
                    
                    # Choose a start vertex for this path
                    path = find_longest_path_greedy(adj, start, used_edges)
                    if len(path) > 1:
                        paths.append(path)
            
            # Check if all edges are covered
            if len(used_edges) == total_edges and len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
            
            if best_count <= 10:  # Good enough for small graphs
                break
        
        # If greedy didn't cover all edges, try a different approach
        if best_paths is None or len(best_paths) >= best_count:
            # Fallback: edge-by-edge but try to merge
            used_edges = set()
            paths = []
            
            for e in edges:
                if time_left() < 1.0:
                    break
                if e in used_edges:
                    continue
                
                # Try to extend existing paths or create new one
                u, v = e
                extended = False
                
                for i, path in enumerate(paths):
                    # Try to prepend u
                    if path[0] == v:
                        new_path = [u] + path
                        # Check simplicity
                        if len(set(new_path)) == len(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                    # Try to append u
                    if path[-1] == v:
                        new_path = path + [u]
                        if len(set(new_path)) == len(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                    # Try to prepend v
                    if path[0] == u:
                        new_path = [v] + path
                        if len(set(new_path)) == len(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                    # Try to append v
                    if path[-1] == u:
                        new_path = path + [v]
                        if len(set(new_path)) == len(new_path):
                            paths[i] = new_path
                            used_edges.add(e)
                            extended = True
                            break
                
                if not extended:
                    paths.append([u, v])
                    used_edges.add(e)
            
            if len(used_edges) == total_edges:
                if best_paths is None or len(paths) < best_count:
                    best_paths = [list(p) for p in paths]
        
        return best_paths if best_paths is not None else [[list(e)] for e in edges]
    
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time_left() < 3.0:
            # Not enough time, use simple fallback
            edges = instance(*spec)
            results.append([[list(e)] for e in edges])
            continue
        
        edges = instance(*spec)
        # Try the matching-based approach first, then fall back to greedy
        paths = decompose_graph_matching(edges)
        if paths is None or len(paths) > len(edges):
            paths = decompose_graph(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
