def construct():
    # Define the lattice neighbors
    # (a, b) neighbors: (a+1, b), (a, b+1), (a-1, b+1), (a-1, b), (a, b-1), (a+1, b-1)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Candidate pool: points within a certain radius
    # We need 32 points. A hexagon of radius 4 has 37 points.
    # So radius 4 is sufficient.
    candidates = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            # Check if distance is small enough
            # a^2 + ab + b^2 <= 16 (radius 4)
            if a*a + a*b + b*b <= 16:
                candidates.append((a, b))
                
    selected = set()
    
    for _ in range(32):
        best_point = None
        max_connections = -1
        for p in candidates:
            if p in selected:
                continue
            # Count neighbors in selected
            connections = 0
            for da, db in neighbors:
                if (p[0]+da, p[1]+db) in selected:
                    connections += 1
            if connections > max_connections:
                max_connections = connections
                best_point = p
        if best_point is not None:
            selected.add(best_point)
            
    return [[a, b] for a, b in selected]
