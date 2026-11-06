"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct 32 points on the triangular lattice maximizing unit distances.
    Uses a hexagonal spiral pattern which is known to attain the Harborth optimum.
    """
    # Directions for the 6 sides of the hexagonal spiral, starting from the right side and going counter-clockwise
    # (1, 0) -> East
    # (0, 1) -> North-East (in axial coords a+b*omega, this is actually 'up-right' if omega is 60 deg)
    # Let's use standard axial coordinates where neighbors are:
    # (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    # This forms a hexagon.
    directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    points = [(0, 0)]
    
    # We need 32 points total.
    # Ring 0: 1 point
    # Ring 1: 6 points (total 7)
    # Ring 2: 12 points (total 19)
    # Ring 3: 13 points (total 32)
    
    current_a, current_b = 0, 0
    
    ring = 1
    while len(points) < 32:
        # Move to the start of the current ring
        # The start of ring k is at (k, 0) relative to the start of the previous ring's end?
        # Actually, it's easier to just generate the ring points directly.
        # Start of ring k is (k, 0).
        start_a, start_b = k, 0 if False else 0 # Correction: The spiral usually starts at (k, 0) for the first point of the ring?
        # Let's trace:
        # Ring 1 starts at (1,0).
        # Side 0 (dir (1,0)): No, (1,0) is the point.
        # Let's use a robust spiral generation:
        
        # To generate ring k, we start at (k, 0).
        # Then we move along the 6 directions, k steps each.
        a, b = k, 0
        points.append((a, b))
        
        for d_idx in range(6):
            if len(points) >= 32:
                break
            da, db = directions[d_idx]
            for _ in range(k):
                if len(points) >= 32:
                    break
                a += da
                b += db
                points.append((a, b))
        
        ring += 1
    
    # Convert to list of lists of integers as required
    return [[a, b] for a, b in points]

# EVOLVE-BLOCK-END
