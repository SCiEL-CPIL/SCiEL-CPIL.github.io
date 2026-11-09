    def find_path_cover_greedy_improved(edges, n):
        if not edges:
            return []
        
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        paths = []
        
        best_paths = None
        best_count = float('inf')
        
        # Limit restarts based on time
        max_restarts = 100
        for attempt in range(max_restarts):
            if time.time() > DEADLINE:
                break
                
            curr_remaining = set(remaining_edges)
            curr_paths = []
            
            while curr_remaining:
                # Pick a random edge
                start_edge = next(iter(curr_remaining))
                u, v = start_edge
                
                # Random direction
                if Random(attempt).random() < 0.5:
                    start, end = u, v
                else:
                    start, end = v, u
                    
                path = [start]
                visited = {start}
                current = start
                
                curr_remaining.discard(start_edge)
                
                while True:
                    # Find candidates
                    candidates = []
                    for nb in adj[current]:
                        if nb in visited:
                            continue
                        e = tuple(sorted((current, nb)))
                        if e in curr_remaining:
                            candidates.append((nb, e))
                            
                    if not candidates:
                        break
                    
                    # Heuristic: Min degree in remaining graph
                    # Calculate remaining degree for each candidate
                    # This is O(E) per step, which is slow.
                    # Approximate: use static degree?
                    # Let's just pick random for speed, or min static degree.
                    
                    # Pick random candidate
                    nb, e = Random(attempt).choice(candidates)
                    
                    curr_remaining.discard(e)
                    path.append(nb)
                    visited.add(nb)
                    current = nb
                
                curr_paths.append(path)
            
            if len(curr_paths) < best_count:
                best_count = len(curr_paths)
                best_paths = curr_paths
                
        return best_paths if best_paths else []
