def construct():
    from random import Random
    import time

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, trials=300, time_limit=30):
        if not edges:
            return []
        
        # Build adjacency list
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        initial_edges = set(tuple(sorted(e)) for e in edges)
        best_paths = None
        best_count = float('inf')
        
        start_time = time.time()
        
        for t in range(trials):
            if time.time() - start_time > time_limit:
                break
                
            remaining_edges = set(initial_edges)
            paths = []
            
            # We need a deterministic but varied source of randomness for each trial
            # Using a separate Random instance for each trial ensures reproducibility
            trial_rng = Random(12345 + t)
            
            while remaining_edges:
                # Find active vertices
                active_vertices = []
                for v in adj:
                    if any(tuple(sorted((v, u))) in remaining_edges for u in adj.get(v, [])):
                        active_vertices.append(v)
                
                if not active_vertices:
                    break
                    
                # Select start vertex
                # Strategy: Mix of min-degree and random
                if trial_rng.random() < 0.5:
                    def get_deg(v):
                        return sum(1 for u in adj.get(v, []) if tuple(sorted((v, u))) in remaining_edges)
                    # Sort by degree, then by vertex id for stability
                    active_vertices.sort(key=lambda v: (get_deg(v), v))
                    start = active_vertices[0]
                else:
                    start = trial_rng.choice(active_vertices)
                
                # Greedy walk
                path = [start]
                used_vertices = {start}
                current = start
                
                while True:
                    neighbors = []
                    for u in adj.get(current, []):
                        if u in used_vertices:
                            continue
                        e = tuple(sorted((current, u)))
                        if e in remaining_edges:
                            neighbors.append(u)
                    
                    if not neighbors:
                        break
                    
                    # Calculate remaining degrees for neighbors
                    def sim_deg(v):
                        count = 0
                        for x in adj.get(v, []):
                            if x in used_vertices:
                                continue
                            e = tuple(sorted((v, x)))
                            if e in remaining_edges:
                                count += 1
                        return count
                    
                    # Find min degree
                    min_deg = min(sim_deg(v) for v in neighbors)
                    candidates = [v for v in neighbors if sim_deg(v) == min_deg]
                    
                    # Random choice among candidates to diversify trials
                    next_v = trial_rng.choice(candidates)
                    
                    path.append(next_v)
                    used_vertices.add(next_v)
                    remaining_edges.remove(tuple(sorted((current, next_v))))
                    current = next_v
                
                if path:
                    paths.append(path)
            
            # Check if this trial produced a better result
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
                
        return best_paths if best_paths else []

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges, trials=500, time_limit=30)
        results.append(paths)
    
    return results
