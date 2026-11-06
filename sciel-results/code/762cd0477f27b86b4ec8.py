def construct():
    """
    Constructs 32 points on the triangular lattice maximizing the number of unit distances.
    Uses a hexagonal spiral (sorted by distance from origin) to form a compact cluster.
    """
    # Generate candidate points in a square grid large enough to contain 32 points
    # The 32nd point in a hexagonal spiral is at distance 3 from the origin.
    # A range of -10 to 10 is sufficient.
    candidates = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            candidates.append((a, b))
    
    # Sort by squared Euclidean distance from the origin
    # Distance squared in axial coordinates (a, b) is a^2 + a*b + b^2
    candidates.sort(key=lambda p: p[0]**2 + p[0]*p[1] + p[1]**2)
    
    # Select the 32 closest points
    selected_points = candidates[:32]
    
    # Return as list of lists [a, b]
    return [[a, b] for a, b in selected_points]
