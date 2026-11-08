<<<<<<< SEARCH
    d = math.sqrt(3)  # center-to-center distance for adjacent unit hexagons
    
    # Base positions: center + 6 first ring
    base = [(0.0, 0.0)]
    for i in range(6):
        a = i * math.pi / 3
        base.append((d * math.cos(a), d * math.sin(a)))
    
    # Second ring: 12 positions
    second_ring = []
    for i in range(6):
        a = i * math.pi / 3
        second_ring.append((2 * d * math.cos(a), 2 * d * math.sin(a)))
    for i in range(6):
        a = i * math.pi / 3 + math.pi / 6
        second_ring.append((d * math.sqrt(3) * math.cos(a), d * math.sqrt(3) * math.sin(a)))
=======
    # Use a slightly larger distance to ensure strict non-overlap (disjointness)
    # and avoid floating point precision issues causing "overlap" errors.
    gap_factor = 1.01
    d = math.sqrt(3) * gap_factor  # center-to-center distance for adjacent unit hexagons
    
    # Base positions: center + 6 first ring
    base = [(0.0, 0.0)]
    for i in range(6):
        a = i * math.pi / 3
        base.append((d * math.cos(a), d * math.sin(a)))
    
    # Second ring: 12 positions
    second_ring = []
    for i in range(6):
        a = i * math.pi / 3
        second_ring.append((2 * d * math.cos(a), 2 * d * math.sin(a)))
    for i in range(6):
        a = i * math.pi / 3 + math.pi / 6
        second_ring.append((d * math.sqrt(3) * math.cos(a), d * math.sqrt(3) * math.sin(a)))
>>>>>>> REPLACE
