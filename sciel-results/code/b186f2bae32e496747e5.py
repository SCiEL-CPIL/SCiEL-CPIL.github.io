def construct():
    import time
    start_time = time.time()
    
    # Candidate pool: points within a radius that contains at least 32 points
    # Radius 4 has 37 points.
    candidates = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            if a*a + a*b + b*b <= 16:
                candidates.append((a, b))
                
    # We need to select 32 points.
    # Greedy: pick point that adds most edges to current set.
    selected = set()
    
    # Start with (0,0) to break symmetry
    selected.add((0,0))
    
    for _ in range(31):
        if time.time() - start_time > 140:
            break
            
        best_point = None
        max_edges = -1
        
        for p in candidates:
            if p in selected:
                continue
                
            # Count edges to selected points
            edges = 0
            for s in selected:
                da = p[0] - s[0]
                db = p[1] - s[1]
                if da*da + da*db + db*db == 1:
                    edges += 1
                    
            if edges > max_edges:
                max_edges = edges
                best_point = p
                
        if best_point:
            selected.add(best_point)
            
    # Convert to list of lists
    result = [[a, b] for a, b in selected]
    return result
