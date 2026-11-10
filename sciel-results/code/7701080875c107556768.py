# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_12():
    """
    Constructs a tight packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    n = 12
    h = np.sqrt(3)

    # Honeycomb lattice arrangement:
    # Row 1 (y = 0): 4 hexagons
    # Row 2 (y = h): 4 hexagons (offset by 1.5)
    # Row 3 (y = 2h): 4 hexagons
    inner_hex_data = np.array([
        [0.0, 0.0, 0],       # Row 1, col 0
        [3.0, 0.0, 0],       # Row 1, col 1
        [6.0, 0.0, 0],       # Row 1, col 2
        [9.0, 0.0, 0],       # Row 1, col 3
        [1.5, h, 0],         # Row 2, col 0
        [4.5, h, 0],         # Row 2, col 1
        [7.5, h, 0],         # Row 2, col 2
        [10.5, h, 0],        # Row 2, col 3
        [0.0, 2*h, 0],       # Row 3, col 0
        [3.0, 2*h, 0],       # Row 3, col 1
        [6.0, 2*h, 0],       # Row 3, col 2
        [9.0, 2*h, 0],       # Row 3, col 3
    ])

    # Center the configuration
    cx = np.mean(inner_hex_data[:, 0])
    cy = np.mean(inner_hex_data[:, 1])
    inner_hex_data[:, 0] -= cx
    inner_hex_data[:, 1] -= cy

    # Compute required outer hexagon side length
    # For a regular hexagon with flat-top orientation (angle=0), 
    # the vertices are at (d*cos(60*i), d*sin(60*i)) for i=0..5
    # The hexagon contains point (x,y) if for all 6 edges, the point is on the correct side
    # 
    # A regular hexagon centered at origin with side d and angle 0 has vertices:
    # (d, 0), (d/2, d*sqrt(3)/2), (-d/2, d*sqrt(3)/2), (-d, 0), (-d/2, -d*sqrt(3)/2), (d/2, -d*sqrt(3)/2)
    # 
    # The 6 edges are defined by 6 half-planes:
    # Edge 0: x <= d
    # Edge 1: x/2 + y*sqrt(3)/2 <= d
    # Edge 2: -x/2 + y*sqrt(3)/2 <= d
    # Edge 3: -x <= d
    # Edge 4: -x/2 - y*sqrt(3)/2 <= d
    # Edge 5: x/2 - y*sqrt(3)/2 <= d
    #
    # So for a point (x,y) to be inside, we need:
    # |x| <= d
    # |x/2 + y*sqrt(3)/2| <= d
    # |x/2 - y*sqrt(3)/2| <= d
    # i.e., max(|x|, |x/2 + y*sqrt(3)/2|, |x/2 - y*sqrt(3)/2|) <= d
    #
    # But we also need to account for the hexagon's own extent (side 1).
    # Each inner hexagon has vertices at distance 1 from its center.
    # So for each inner hexagon center (cx_i, cy_i), we need to find the maximum 
    # of max(|x|, |x/2 + y*sqrt(3)/2|, |x/2 - y*sqrt(3)/2|) over all 6 vertices of that hexagon.
    # The outer hexagon side length must be at least the maximum over all inner hexagons.

    d_max = 0
    for i in range(n):
        cx_i = inner_hex_data[i, 0]
        cy_i = inner_hex_data[i, 1]
        # 6 vertices of unit hexagon at (cx_i, cy_i) with angle 0
        for j in range(6):
            angle = np.radians(60 * j)
            vx = cx_i + np.cos(angle)
            vy = cy_i + np.sin(angle)
            # Compute the "hexagonal norm" of this vertex
            val1 = abs(vx)
            val2 = abs(vx / 2 + vy * np.sqrt(3) / 2)
            val3 = abs(vx / 2 - vy * np.sqrt(3) / 2)
            d_needed = max(val1, val2, val3)
            d_max = max(d_max, d_needed)

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = d_max

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
