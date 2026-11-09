import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Coordinates for 11 hexagons in a honeycomb pattern
    # Axial coordinates (q, r)
    # 7 central hexes (norm 0 and 1)
    # 4 side hexes from norm 2 (avoiding corners to keep max radius low)
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-1, 2), (-2, 1)
    ]
    
    inner = []
    for q, r in axial_coords:
        # Convert axial to Cartesian for flat-top hexagons
        # x = sqrt(3) * (q + r/2)
        # y = 1.5 * r
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    # The outer hexagon must contain these.
    # Max distance from center is 4.0 (at the vertices of the side hexes).
    # We align the outer hexagon's vertices with these max points.
    # The max points are at angles -30, 30, 90, 150, etc.
    # A regular hexagon with side S=4 has circumradius 4.
    # If we rotate it by 30 degrees, its vertices are at 30, 90, 150, 210, 270, 330.
    # This matches the directions of the farthest points.
    
    return inner, (0.0, 0.0), 4.0, 30.0
# EVOLVE-BLOCK-END
