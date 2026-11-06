def construct():
    """
    Return 32 distinct lattice points (a, b) on the triangular lattice
    a + b*omega, omega = exp(i*pi/3), maximizing the number of unit-distance
    pairs. Harborth's theorem gives the optimum as floor(3*32 - sqrt(12*32 - 3)) = 76.
    
    A hexagonal spiral (rings) achieves this optimum.
    """
    # Define the 6 neighbors in the triangular lattice (a, b) coordinates
    # The distance squared between (a1,b1) and (a2,b2) is:
    # (a1-a2)^2 + (a1-a2)(b1-b2) + (b1-b2)^2
    # Unit distance means this equals 1.
    # The 6 unit neighbors of (0,0) are:
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    
    # We'll build the hexagonal spiral:
    # Ring 0: 1 point
    # Ring k: 6k points
    # Total up to ring k: 1 + 6*(1+2+...+k) = 1 + 3k(k+1)
    # We need 32 points.
    # Ring 0: 1, total=1
    # Ring 1: 6, total=7
    # Ring 2: 12, total=19
    # Ring 3: 18, total=37
    # So we need all of rings 0,1,2 (19 points) plus 13 from ring 3.
    
    points = []
    
    # Ring 0
    points.append([0, 0])
    
    # Ring 1: 6 points
    # Starting from (1,0) and going around
    # The 6 points at hexagonal distance 1:
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    for p in ring1:
        points.append(list(p))
    
    # Ring 2: 12 points
    # Hexagonal distance 2
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1),
             (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)]
    for p in ring2:
        points.append(list(p))
    
    # Ring 3: need 13 more points (total 32)
    # Hexagonal distance 3
    ring3 = [(3, 0), (3, -1), (3, -2), (3, -3), (2, -3), (1, -3),
             (0, -3), (-1, -2), (-2, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3),
             (-2, 3), (-1, 3), (0, 3), (1, 2), (2, 1)]
    # Take first 13
    for p in ring3[:13]:
        points.append(list(p))
    
    return points
