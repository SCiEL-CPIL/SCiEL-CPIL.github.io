import random
import time
import sys

def construct():
    """
    Decomposes three fixed connected graphs into edge-disjoint simple paths.
    Minimizes the total number of paths using a greedy heuristic with
    strategic vertex ordering and local improvement attempts.
    """
    
    def instance_graph(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def greedy_decompose(edges, n, time_budget):
        """
        Greedy path decomposition with smart edge selection.
        Tries to minimize path count by preferring edges that connect
        high-degree vertices and avoiding premature path termination.
        """
        if not edges:
            return []
        
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Precompute degree
        degree = {i: len(adj[i]) for i in range(n)}
        
        best_paths = None
        best_count = float('inf')
        start_time = time.time()
        
        # Multiple restarts with different random seeds
        rng = random.Random(42)
        
        iteration = 0
        while time.time() - start_time < time_budget:
            iteration += 1
            remaining = set(edges)
            paths = []
            
            # Sort remaining edges by some heuristic
            # Prefer edges between high-degree vertices
            def edge_priority(edge):
                u, v = edge
                # Higher priority for edges connecting high-degree vertices
                # Also consider how many "options" each vertex has in remaining graph
                return -(degree[u] + degree[v])
            
            # Use a priority queue-like approach by repeatedly picking best edge
            # For efficiency, we'll use a simpler strategy: random with bias
            
            while remaining:
                if time.time() - start_time >= time_budget:
                    break
                    
                # Choose start edge with bias toward high-degree vertices
                # Sample a few candidates and pick the best
                candidates = []
                sample_size = min(10, len(remaining))
                sampled = random.sample(list(remaining), sample_size)
                
                for edge in sampled:
                    u, v = edge
                    # Count available extensions for u and v in remaining graph
                    # This is expensive, so use degree as proxy
                    candidates.append((-(degree[u] + degree[v]), edge))
                
                candidates.sort()
                u, v = candidates[0][1]
                
                path = [u, v]
                remaining.discard((u, v))
                
                # Track which vertices are in the path for O(1) lookup
                path_set = {u, v}
                
                # Greedily extend the path
                while True:
                    extended = False
                    
                    # Try extending from both ends, prefer the one with more options
                    # Count options for u
                    options_u = []
                    for w in adj[u]:
                        if (u, w) in remaining or (w, u) in remaining:
                            if w not in path_set:
                                options_u.append(w)
                    
                    # Count options for v
                    options_v = []
                    for w in adj[v]:
                        if (v, w) in remaining or (w, v) in remaining:
                            if w not in path_set:
                                options_v.append(w)
                    
                    # Choose which end to extend
                    if len(options_u) > len(options_v):
                        # Extend from left (u)
                        if options_u:
                            # Prefer vertex with lower degree (more "end-like")
                            # or random
                            random.shuffle(options_u)
                            for w in options_u:
                                edge_key = tuple(sorted((u, w)))
                                if edge_key in remaining:
                                    path.insert(0, w)
                                    path_set.add(w)
                                    remaining.discard(edge_key)
                                    u = w
                                    extended = True
                                    break
                    elif len(options_v) > 0:
                        # Extend from right (v)
                        random.shuffle(options_v)
                        for w in options_v:
                            edge_key = tuple(sorted((v, w)))
                            if edge_key in remaining:
                                path.append(w)
                                path_set.add(w)
                                remaining.discard(edge_key)
                                v = w
                                extended = True
                                break
                    elif options_u:
                        # Only options on left
                        random.shuffle(options_u)
                        for w in options_u:
                            edge_key = tuple(sorted((u, w)))
                            if edge_key in remaining:
                                path.insert(0, w)
                                path_set.add(w)
                                remaining.discard(edge_key)
                                u = w
                                extended = True
                                break
                    
                    if not extended:
                        break
                
                paths.append(path)
            
            if paths and len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
                
                # Check if we can stop early
                # Lower bound: at least 1 path if graph is connected and has edges
                # Better lower bound: number of odd-degree vertices / 2 + 1 (for Eulerian trails)
                # But for simple path decomposition, it's more complex
                if best_count <= 5:  # Reasonable early stop
                    break
        
        if best_paths is None:
            # Fallback: single edges
            return [[u, v] for u, v in edges]
        
        return best_paths
    
    def local_improve(paths, edges, n, time_budget):
        """
        Try to reduce number of paths by merging adjacent paths.
        Two paths can be merged if one ends at a vertex that is the start
        of the other, and the merge creates a valid simple path.
        """
        if not paths:
            return paths
        
        start_time = time.time()
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            new_paths = [p[:] for p in paths]
            
            # Try all pairs of paths
            for i in range(len(new_paths)):
                if time.time() - start_time >= time_budget:
                    break
                for j in range(i + 1, len(new_paths)):
                    if time.time() - start_time >= time_budget:
                        break
                    
                    p1 = new_paths[i]
                    p2 = new_paths[j]
                    
                    # Try merging: p1's end + p2's start, or p1's start + p2's end, etc.
                    # Check if last vertex of p1 is first vertex of p2, or vice versa
                    # Or if last of p1 is last of p2, or first of p1 is first of p2
                    
                    merge_successful = False
                    
                    # Case 1: p1[-1] == p2[0] -> merge p1 + p2[1:]
                    if p1[-1] == p2[0]:
                        merged = p1 + p2[1:]
                        # Check if merged is simple (all vertices unique)
                        if len(merged) == len(set(merged)):
                            # Check all edges are valid
                            valid = True
                            for k in range(len(merged) - 1):
                                a, b = merged[k], merged[k + 1]
                                if b not in adj[a]:
                                    valid = False
                                    break
                            if valid:
                                new_paths[i] = merged
                                del new_paths[j]
                                improved = True
                                merge_successful = True
                                break
                    
                    # Case 2: p1[0] == p2[-1] -> merge p1[1:] + p2
                    if not merge_successful and p1[0] == p2[-1]:
                        merged = p1[1:] + p2
                        if len(merged) == len(set(merged)):
                            valid = True
                            for k in range(len(merged) - 1):
                                a, b = merged[k], merged[k + 1]
                                if b not in adj[a]:
                                    valid = False
                                    break
                            if valid:
                                new_paths[i] = merged
                                del new_paths[j]
                                improved = True
                                merge_successful = True
                                break
                    
                    # Case 3: p1[-1] == p2[-1] -> merge p1 + reversed(p2[:-1])
                    if not merge_successful and p1[-1] == p2[-1]:
                        merged = p1 + list(reversed(p2[:-1]))
                        if len(merged) == len(set(merged)):
                            valid = True
                            for k in range(len(merged) - 1):
                                a, b = merged[k], merged[k + 1]
                                if b not in adj[a]:
                                    valid = False
                                    break
                            if valid:
                                new_paths[i] = merged
                                del new_paths[j]
                                improved = True
                                merge_successful = True
                                break
                    
                    # Case 4: p1[0] == p2[0] -> merge reversed(p1[1:]) + p2
                    if not merge_successful and p1[0] == p2[0]:
                        merged = list(reversed(p1[1:])) + p2
                        if len(merged) == len(set(merged)):
                            valid = True
                            for k in range(len(merged) - 1):
                                a, b = merged[k], merged[k + 1]
                                if b not in adj[a]:
                                    valid = False
                                    break
                            if valid:
                                new_paths[i] = merged
                                del new_paths[j]
                                improved = True
                                merge_successful = True
                                break
            
            if improved:
                paths = new_paths
        
        return paths
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    # Allocate time budget: 140s total, leave 10s margin
    # Distribute proportionally to graph size
    total_weight = sum(spec[0] for spec in specs)
    time_budget_total = 140.0
    
    for n, p, seed in specs:
        edges = instance_graph(n, p, seed)
        t_greedy = time_budget_total * n / total_weight * 0.7  # 70% for greedy
        t_improve = time_budget_total * n / total_weight * 0.3  # 30% for local improvement
        
        # Greedy decomposition
        paths = greedy_decompose(edges, n, t_greedy)
        
        # Local improvement
        if time_budget_total * n / total_weight > 5:  # Only if we have time
            paths = local_improve(paths, edges, n, t_improve)
        
        results.append(paths)
        
    return results

# EVOLVE-BLOCK-START
def construct():
    """
    Decomposes three fixed connected graphs into edge-disjoint simple paths.
    Minimizes the total number of paths using a combination of greedy heuristic
    and local improvement (path merging).
    """
    
    def instance_graph(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def greedy_decompose(edges, n, time_budget):
        """
        Greedy path decomposition with smart edge selection.
        """
        if not edges:
            return []
        
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        degree = {i: len(adj[i]) for i in range(n)}
        
        best_paths = None
        best_count = float('inf')
        start_time = time.time()
        
        rng = random.Random(42)
        
        while time.time() - start_time < time_budget:
            remaining = set(edges)
            paths = []
            
            while remaining:
                if time.time() - start_time >= time_budget:
                    break
                
                # Choose start edge with bias toward high-degree vertices
                candidates = []
                sample_size = min(10, len(remaining))
                sampled = random.sample(list(remaining), sample_size)
                
                for edge in sampled:
                    u, v = edge
                    candidates.append((-(degree[u] + degree[v]), edge))
                
                candidates.sort()
                u, v = candidates[0][1]
                
                path = [u, v]
                remaining.discard((u, v))
                
                path_set = {u, v}
                
                while True:
                    extended = False
                    
                    options_u = [w for w in adj[u] if (u, w) in remaining or (w, u) in remaining if w not in path_set]
                    options_v = [w for w in adj[v] if (v, w) in remaining or (w, v) in remaining if w not in path_set]
                    
                    if len(options_u) >= len(options_v) and options_u:
                        random.shuffle(options_u)
                        for w in options_u:
                            edge_key = tuple(sorted((u, w)))
                            if edge_key in remaining:
                                path.insert(0, w)
                                path_set.add(w)
                                remaining.discard(edge_key)
                                u = w
                                extended = True
                                break
                    elif options_v:
                        random.shuffle(options_v)
                        for w in options_v:
                            edge_key = tuple(sorted((v, w)))
                            if edge_key in remaining:
                                path.append(w)
                                path_set.add(w)
                                remaining.discard(edge_key)
                                v = w
                                extended = True
                                break
                    elif options_u:
                        random.shuffle(options_u)
                        for w in options_u:
                            edge_key = tuple(sorted((u, w)))
                            if edge_key in remaining:
                                path.insert(0, w)
                                path_set.add(w)
                                remaining.discard(edge_key)
                                u = w
                                extended = True
                                break
                    
                    if not extended:
                        break
                
                paths.append(path)
            
            if paths and len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
                if best_count <= 5:
                    break
        
        if best_paths is None:
            return [[u, v] for u, v in edges]
        
        return best_paths
    
    def local_improve(paths, edges, n, time_budget):
        """
        Try to reduce number of paths by merging adjacent paths.
        """
        if not paths:
            return paths
        
        start_time = time.time()
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            new_paths = [p[:] for p in paths]
            
            for i in range(len(new_paths)):
                if time.time() - start_time >= time_budget:
                    break
                for j in range(i + 1, len(new_paths)):
                    if time.time() - start_time >= time_budget:
                        break
                    
                    p1 = new_paths[i]
                    p2 = new_paths[j]
                    
                    merge_successful = False
                    
                    if p1[-1] == p2[0]:
                        merged = p1 + p2[1:]
                        if len(merged) == len(set(merged)):
                            valid = all(b in adj[a] for a, b in zip(merged, merged[1:]))
                            if valid:
                                new_paths[i] = merged
                                del new_paths[j]
                                improved = True
                                merge_successful = True
                                break
                    
                    if not merge_successful and p1[0] == p2[-1]:
                        merged = p1[1:] + p2
                        if len(merged) == len(set(merged)):
                            valid = all(b in adj[a] for a, b in zip(merged, merged[1:]))
                            if valid:
                                new_paths[i] = merged
                                del new_paths[j]
                                improved = True
                                merge_successful = True
                                break
                    
                    if not merge_successful and p1[-1] == p2[-1]:
                        merged = p1 + list(reversed(p2[:-1]))
                        if len(merged) == len(set(merged)):
                            valid = all(b in adj[a] for a, b in zip(merged, merged[1:]))
                            if valid:
                                new_paths[i] = merged
                                del new_paths[j]
                                improved = True
                                merge_successful = True
                                break
                    
                    if not merge_successful and p1[0] == p2[0]:
                        merged = list(reversed(p1[1:])) + p2
                        if len(merged) == len(set(merged)):
                            valid = all(b in adj[a] for a, b in zip(merged, merged[1:]))
                            if valid:
                                new_paths[i] = merged
                                del new_paths[j]
                                improved = True
                                merge_successful = True
                                break
            
            if improved:
                paths = new_paths
        
        return paths
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    total_weight = sum(spec[0] for spec in specs)
    time_budget_total = 140.0
    
    for n, p, seed in specs:
        edges = instance_graph(n, p, seed)
        t_greedy = time_budget_total * n / total_weight * 0.7
        t_improve = time_budget_total * n / total_weight * 0.3
        
        paths = greedy_decompose(edges, n, t_greedy)
        
        if time_budget_total * n / total_weight > 5:
            paths = local_improve(paths, edges, n, t_improve)
        
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
