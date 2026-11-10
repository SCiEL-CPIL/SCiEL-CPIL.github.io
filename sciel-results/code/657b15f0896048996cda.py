"""Program that constructs 32 lattice points achieving the Harborth bound of 76 edges."""

def construct():
    """
    Returns exactly 32 distinct integer pairs [a, b] with |a|, |b| <= 64
    denoting lattice points a + b*omega, omega = exp(i*pi/3), which are at
    mutual distance at least one. Maximizes the number of pairs at distance
    exactly one; Harborth's theorem gives the optimum 76.
    
    Strategy: Start with the r=3 hexagon (37 points, 89 edges) and remove 5
    consecutive boundary points that include 2 corners, losing exactly 14 edges
    to achieve the optimal 76 edges.
    """
    
    # Generate the r=3 hexagon in axial coordinates
    # Hex distance: max(|a|, |b|, |a+b|)
    # The r=3 hexagon contains all points with hex distance <= 3
    
    hex_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a + b)) <= 3:
                hex_points.append((a, b))
    
    # Verify we have 37 points
    assert len(hex_points) == 37, f"Expected 37 points, got {len(hex_points)}"
    
    # The 6 corners of the r=3 hexagon are:
    # (3, 0), (0, 3), (-3, 3), (-3, 0), (0, -3), (3, -3)
    corners = [(3, 0), (0, 3), (-3, 3), (-3, 0), (0, -3), (3, -3)]
    
    # We need to remove 5 consecutive points along the boundary that include 2 corners.
    # Consider the edge from corner (3, 0) to corner (0, 3):
    # Points on this edge: (3, 0), (2, 1), (1, 2), (0, 3)
    # These are 4 points. We need 5 consecutive points including 2 corners.
    # 
    # Let's take: (3, 0), (2, 1), (1, 2), (0, 3), (-1, 3)
    # This includes corners (3, 0) and (0, 3).
    # 
    # Alternatively, consider the edge from (0, 3) to (-3, 3):
    # Points: (0, 3), (-1, 3), (-2, 3), (-3, 3)
    # 
    # Let's remove: (3, 0), (2, 1), (1, 2), (0, 3), (-1, 3)
    # This block of 5 consecutive boundary points includes 2 corners: (3, 0) and (0, 3).
    
    remove_set = {(3, 0), (2, 1), (1, 2), (0, 3), (-1, 3)}
    
    # Verify all points to remove are in the hexagon
    for p in remove_set:
        assert p in hex_points, f"Point {p} not in hexagon"
    
    # Remove these 5 points
    final_points = [p for p in hex_points if p not in remove_set]
    
    # Verify we have exactly 32 points
    assert len(final_points) == 32, f"Expected 32 points, got {len(final_points)}"
    
    # Verify all points are distinct
    assert len(set(final_points)) == 32
    
    # Verify |a|, |b| <= 64
    for a, b in final_points:
        assert abs(a) <= 64 and abs(b) <= 64
    
    # Count edges to verify we achieve 76
    point_set = set(final_points)
    # Neighbors in axial coordinates: (a±1,b), (a,b±1), (a+1,b-1), (a-1,b+1)
    neighbor_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    
    edge_count = 0
    for a, b in final_points:
        for da, db in neighbor_offsets:
            if (a + da, b + db) in point_set:
                edge_count += 1
    edge_count //= 2  # Each edge counted twice
    
    # The optimal value is 76
    # Let's verify: if not 76, try a different removal set
    
    if edge_count != 76:
        # Try different removal blocks
        # Edge from (3,0) to (0,3): (3,0), (2,1), (1,2), (0,3)
        # Edge from (0,3) to (-3,3): (0,3), (-1,3), (-2,3), (-3,3)
        # 
        # Option 1: (3,0), (2,1), (1,2), (0,3), (-1,3) - includes corners (3,0) and (0,3)
        # Option 2: (2,1), (1,2), (0,3), (-1,3), (-2,3) - includes corner (0,3) only
        # Option 3: (1,2), (0,3), (-1,3), (-2,3), (-3,3) - includes corners (0,3) and (-3,3)
        # Option 4: (3,-3), (3,-2), (3,-1), (3,0), (2,1) - includes corner (3,-3) only
        # Option 5: (0,-3), (1,-3), (2,-3), (3,-3), (3,-2) - includes corner (0,-3) and (3,-3)
        
        # Let's try Option 3: includes corners (0,3) and (-3,3)
        remove_set_alt = {(1, 2), (0, 3), (-1, 3), (-2, 3), (-3, 3)}
        final_points_alt = [p for p in hex_points if p not in remove_set_alt]
        
        point_set_alt = set(final_points_alt)
        edge_count_alt = 0
        for a, b in final_points_alt:
            for da, db in neighbor_offsets:
                if (a + da, b + db) in point_set_alt:
                    edge_count_alt += 1
        edge_count_alt //= 2
        
        if edge_count_alt == 76:
            final_points = final_points_alt
            edge_count = 76
        else:
            # Try Option 5: includes corners (0,-3) and (3,-3)
            remove_set_alt2 = {(0, -3), (1, -3), (2, -3), (3, -3), (3, -2)}
            final_points_alt2 = [p for p in hex_points if p not in remove_set_alt2]
            
            point_set_alt2 = set(final_points_alt2)
            edge_count_alt2 = 0
            for a, b in final_points_alt2:
                for da, db in neighbor_offsets:
                    if (a + da, b + db) in point_set_alt2:
                        edge_count_alt2 += 1
            edge_count_alt2 //= 2
            
            if edge_count_alt2 == 76:
                final_points = final_points_alt2
                edge_count = 76
    
    # Convert to list of lists as required
    return [[a, b] for a, b in final_points]
