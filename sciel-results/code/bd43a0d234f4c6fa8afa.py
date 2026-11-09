# EVOLVE-BLOCK-START
import numpy as np
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a lattice-based packing with outer hexagon side length 4.
    """
    start_time = time.time()
    
    # Lattice basis vectors for unit hexagon centers (spacing sqrt(3))
    # e1 = (sqrt(3), 0)
    # e2 = (sqrt(3)/2, 1.5)
    # Center (i, j) -> (sqrt(3)*(i + j/2), 1.5*j)
    
    def get_center(i, j):
        x = np.sqrt(3) * (i + j / 2.0)
        y = 1.5 * j
        return [x, y, 0]

    # 7 inner points (hexagon of side 1 in lattice units)
    centers = []
    centers.append(get_center(0, 0))
    centers.append(get_center(1, 0))
    centers.append(get_center(0, 1))
    centers.append(get_center(-1, 1))
    centers.append(get_center(-1, 0))
    centers.append(get_center(0, -1))
    centers.append(get_center(1, -1))
    
    # 5 points from the next ring (distance 3.0 from center, fitting in D=3)
    # These are (1,1), (-1,2), (-2,1), (-2,-1), (1,-2)
    centers.append(get_center(1, 1))
    centers.append(get_center(-1, 2))
    centers.append(get_center(-2, 1))
    centers.append(get_center(-2, -1))
    centers.append(get_center(1, -2))
    
    inner_hex_data = np.array(centers)
    
    # Outer hexagon
    # Side length s = 4
    # Centered at (0,0), angle 0
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = 4.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
