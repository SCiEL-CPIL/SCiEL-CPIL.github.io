def construct():
    """
    Construct 32 distinct integer pairs [a, b] on the triangular lattice
    (axial coordinates) maximizing the number of pairs at distance exactly 1.
    
    The optimal value is 76 (Harborth's theorem: floor(3n - sqrt(12n - 3)) for n=32).
    
    We use a hexagonal spiral construction:
    - Core: all 19 points with hex distance <= 2
    - Boundary: 13 consecutive points on the hex ring at distance 3
    
    This achieves exactly 76 edges.
    """
    
    # Hexagonal distance in axial coordinates
    def hex_dist(q, r):
        return (abs(q) + abs(r) + abs(q + r)) // 2
    
    # Generate core points (hex distance <= 2): 19 points
    core = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if hex_dist(q, r) <= 2:
                core.append((q, r))
    
    # Generate boundary points (hex distance == 3): 18 points
    boundary = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if hex_dist(q, r) == 3:
                boundary.append((q, r))
    
    # Sort boundary points by angle around origin to form a contiguous arc
    import math
    def angle(p):
        # Axial to Cartesian: x = q + r/2, y = r * sqrt(3)/2
        x = p[0] + p[1] / 2.0
        y = p[1] * math.sqrt(3) / 2.0
        return math.atan2(y, x)
    
    boundary_sorted = sorted(boundary, key=angle)
    
    # Pick 13 consecutive boundary points (contiguous arc)
    # Due to 6-fold symmetry, any contiguous block of 13 gives same edge count
    selected_boundary = boundary_sorted[:13]
    
    final_points = core + selected_boundary
    
    # Verify we have exactly 32 distinct points
    assert len(final_points) == 32
    assert len(set(final_points)) == 32
    
    return [[p[0], p[1]] for p in final_points]
