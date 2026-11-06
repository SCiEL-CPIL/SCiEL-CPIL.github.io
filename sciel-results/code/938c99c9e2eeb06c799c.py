import time

def construct():
    """
    Constructs 32 points on the triangular lattice maximizing the number of edges.
    Uses the hexagonal spiral ordering, which is known to attain the Harborth bound.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n_points = 32
    
    # Generate points in hexagonal spiral order
    points = [(0, 0)]
    
    # Directions for the 6 sides of the hexagon
    # We traverse the perimeter of the hexagon of radius r
    # Start at (r, 0)
    # Side 1: move (-1, 1) r times
    # Side 2: move (-1, 0) r times
    # Side 3: move (0, -1) r times
    # Side 4: move (1, -1) r times
    # Side 5: move (1, 0) r times
    # Side 6: move (0, 1) r times
    
    directions = [
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        (1, 0),
        (0, 1)
    ]
    
    r = 1
    while len(points) < n_points:
        # Start of the ring r is at (r, 0)
        # But we need to generate the points of the ring
        # The ring r consists of 6r points
        # We start from the previous end point?
        # Actually, it's easier to just generate the coordinates for ring r
        
        # The points in ring r are at hex-distance r.
        # We can generate them by walking the perimeter.
        # Start at (r, 0)
        a, b = r, 0
        for side in range(6):
            da, db = directions[side]
            for step in range(r):
                # Add the current point (a,b)
                if len(points) >= n_points:
                    break
                points.append((a, b))
                a += da
                b += db
            if len(points) >= n_points:
                break
        r += 1
        
    # Convert to list of lists
    result = [[a, b] for a, b in points[:n_points]]
    
    return result
