"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Generate a hexagon of radius 3 (side length 3)
    # Points (a, b) such that |a| <= 3, |b| <= 3, |a - b| <= 3
    points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if abs(a) <= 3 and abs(b) <= 3 and abs(a - b) <= 3:
                points.append([a, b])
    
    # We have 37 points. We need 32.
    # Remove 5 of the 6 corners to minimize edge loss.
    # Corners are: (3,0), (0,3), (-3,3), (-3,0), (0,-3), (3,-3)
    corners = [
        [3, 0],
        [0, 3],
        [-3, 3],
        [-3, 0],
        [0, -3],
        [3, -3]
    ]
    
    # Remove the first 5 corners
    for c in corners[:5]:
        if c in points:
            points.remove(c)
            
    # Ensure we have exactly 32 points
    if len(points) != 32:
        # Fallback: just take the first 32
        points = points[:32]
        
    return points
# EVOLVE-BLOCK-END
