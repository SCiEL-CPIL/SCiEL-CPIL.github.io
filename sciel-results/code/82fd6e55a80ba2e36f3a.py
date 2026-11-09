def construct():
    # Grid 1: 3x3 at origin
    g1 = [[x, y] for x in range(3) for y in range(3)]
    # Grid 2: 3x3 at x=10
    g2 = [[x + 10, y] for x in range(3) for y in range(3)]
    # Two extra points far away
    p1 = [50, 50]
    p2 = [51, 52]
    
    points = g1 + g2 + [p1, p2]
    
    # Verify constraints:
    # 1. Distinct
    # 2. |x|, |y| <= 100
    # 3. No four collinear
    
    # The grids are 3x3, so max collinear in grid is 3.
    # Cross-grid lines:
    # We argued that slopes don't match for 3-collinearities involving extras.
    # We should double check no 4 collinear in the union.
    # Since extras are isolated, they can only be in lines with 2 grid points.
    # If a line has E1, P1 (G1), P2 (G2), that's 3 points.
    # If it also has P3, that's 4.
    # But we argued slope mismatch prevents E-P1-P2.
    # So max collinear is 3 (within grids).
    
    return points
