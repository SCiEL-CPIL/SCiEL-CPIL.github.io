<<<<<<< SEARCH
    # 12 positions in a compact hexagonal cluster
    # Let's use: a central hexagon, 6 around it (dist 2 in hex units... wait)
    # Adjacent hexagons in hex grid are at distance sqrt(3) in Cartesian for side-1 hexagons
    # No wait: for regular hexagons of side 1 tiling, centers are at distance 2*apothem = 2*(sqrt(3)/2) = sqrt(3)
    
    # Let me place 12 hexagons in a hexagonal pattern:
    # Layer 0: 1 (center)
    # Layer 1: 6 (at distance sqrt(3))
    # Layer 2: 5 more to make 12
    # Total: 1+6+5 = 12
    
    # Better: use a known compact shape. Let me use axial coords:
    # (0,0) - center
    # (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1) - 6 neighbors
    # (2,0),(-2,0),(0,2),(0,-2),(1,-2) - 5 more
    # That's 12
    
    axial_positions = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (1, -2)
    ]
    
    # Convert to Cartesian (for pointy-top hexagons with side 1)
    centers = []
    for q, r in axial_positions:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Center the cluster
    center_offset = centers.mean(axis=0)
    centers = centers - center_offset
=======
    # 12 positions in a compact hexagonal cluster
    # Using axial coordinates for a valid non-overlapping configuration
    # (0,0) - center
    # (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1) - 6 neighbors
    # (2,0),(-2,0),(0,2),(0,-2),(1,-2) - 5 more
    # Total 12
    
    axial_positions = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (1, -2)
    ]
    
    # Convert to Cartesian (for pointy-top hexagons with side 1)
    centers = []
    for q, r in axial_positions:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Center the cluster
    center_offset = centers.mean(axis=0)
    centers = centers - center_offset
    
    # Scale up slightly to ensure strict non-overlap (avoid boundary touching)
    # The minimum distance is sqrt(3). Scaling by 1.001 ensures gap.
    centers = centers * 1.001
>>>>>>> REPLACE
