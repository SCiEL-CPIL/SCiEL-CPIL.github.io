# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a hexagonal lattice arrangement aligned with the outer hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    # Lattice basis vectors for unit hexagons (aligned)
    # Distance between centers is sqrt(3)
    e1 = np.array([np.sqrt(3), 0])
    e2 = np.array([np.sqrt(3)/2, 1.5])
    
    # 7 inner hexagons: center and 6 neighbors (dist sq 3)
    centers = [
        np.array([0, 0]),
        e1,
        e2,
        -e1 + e2,
        -e1,
        -e2,
        e1 - e2
    ]
    
    # 5 outer hexagons from the next ring (dist sq 9)
    # These are at e1+e2, e1-e2 (wait, e1-e2 is dist sq 3? No, e1-e2 is a neighbor)
    # The dist sq 9 points are:
    # e1+e2, e1-e2? No.
    # q,r coords for dist sq 9 (q^2+qr+r^2=3):
    # (1,1) -> e1+e2
    # (1,-2) -> e1-2e2
    # (-1,-1) -> -e1-e2
    # (-1,2) -> -e1+2e2
    # (2,-1) -> 2e1-e2
    # (-2,1) -> -2e1+e2
    
    outer_points = [
        e1 + e2,
        e1 - 2*e2,
        -e1 - e2,
        -e1 + 2*e2,
        2*e1 - e2
    ]
    
    centers.extend(outer_points)
    
    inner_hex_data = np.zeros((12, 3))
    for i, c in enumerate(centers):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = 4.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
