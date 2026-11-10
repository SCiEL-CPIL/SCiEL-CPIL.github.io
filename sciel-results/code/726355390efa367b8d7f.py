def construct():
    # Generate a hexagonal cluster on the triangular lattice
    # Points are (a, b) representing a + b*omega, omega = exp(i*pi/3)
    # Distance squared between (a1,b1) and (a2,b2) is (da)^2 + da*db + (db)^2
    # Unit distance pairs have (da,db) in {(1,0),(0,1),(1,-1),(-1,0),(0,-1),(-1,1)}
    
    # Generate a hexagonal shape with 32 points
    # Hexagon of radius 2 has 19 points, we need 13 more
    
    # Let's generate points in a hexagonal pattern
    # Using axial coordinates: for each row y, the x range is [-r, r] adjusted
    
    # Actually, let me just generate a compact region and pick 32 points
    # that maximize edges
    
    # Generate candidate points in a hexagonal region
    candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            candidates.append((a, b))
    
    # We need to select 32 points maximizing unit-distance pairs
    # This is NP-hard in general, but for 32 points on a lattice we can use
    # a greedy approach or known optimal configurations
    
    # Known optimal: hexagonal cluster
    # Let's build a hexagon of radius 2 (19 points) and add 13 more
    
    # Hexagon of radius r in axial coords (q, r) where distance is max(|q|,|r|,|q+r|)
    # But our coords are (a, b) with distance formula (da)^2 + da*db + (db)^2
    # The 6 neighbors of (0,0) are (1,0),(0,1),(1,-1),(-1,0),(0,-1),(-1,1)
    
    # Let me just generate a hexagonal shape directly
    # For a hexagonal cluster, I'll use the fact that in the triangular lattice,
    # a "hexagon" can be defined as points within a certain Chebyshev-like distance
    
    # Let me try: all (a,b) with max(|a|, |b|, |a+b|) <= 2 gives 19 points
    hex_pts = set()
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) <= 2:
                hex_pts.add((a, b))
    
    # That should give 19 points
    # Now add 13 more points around the perimeter
    
    # Let me just generate a larger hexagon and pick 32
    # Hexagon of radius 3 has 37 points, so I need to remove 5
    
    hex3 = set()
    for a in range(-4, 5):
        for b in range(-4, 5):
            if max(abs(a), abs(b), abs(a+b)) <= 3:
                hex3.add((a, b))
    
    # hex3 should have 37 points
    # Remove 5 points to get 32, choosing removals that minimize edge loss
    
    # First, let's count edges in hex3
    def count_edges(pts):
        pts_list = list(pts)
        edges = 0
        neighbor_diffs = [(1,0),(0,1),(1,-1),(-1,0),(0,-1),(-1,1)]
        for i, p in enumerate(pts_list):
            for da, db in neighbor_diffs:
                q = (p[0]+da, p[1]+db)
                if q in pts and p < q:  # avoid double counting
                    edges += 1
        return edges
    
    # Actually let me just directly construct the optimal 32-point hexagonal cluster
    # by starting with the full radius-3 hexagon (37 pts) and removing 5 corner points
    
    # The corners of the radius-3 hexagon are at distance 3 from center
    # Removing corner points loses fewer edges than removing edge points
    
    # Let me identify which points to remove
    # Points with max(|a|,|b|,|a+b|) == 3 are on the boundary
    # Corners have 2 coordinates at extremal values
    
    # Let me just try removing 5 points and check edge count
    # Start with all 37, remove points one by one choosing the one that loses fewest edges
    
    current = set(hex3)
    while len(current) > 32:
        best_remove = None
        best_loss = float('inf')
        for p in current:
            # Count how many neighbors p has in current
            neighbors = 0
            neighbor_diffs = [(1,0),(0,1),(1,-1),(-1,0),(0,-1),(-1,1)]
            for da, db in neighbor_diffs:
                q = (p[0]+da, p[1]+db)
                if q in current:
                    neighbors += 1
            if neighbors < best_loss:
                best_loss = neighbors
                best_remove = p
        current.remove(best_remove)
    
    result = [[a, b] for a, b in current]
    return result
