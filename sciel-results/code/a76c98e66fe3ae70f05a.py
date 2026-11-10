# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a hexagonal lattice arrangement: 7 core hexagons + 5 from the second ring (side positions).
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    # Lattice vectors
    sqrt3 = np.sqrt(3)
    u = np.array([sqrt3, 0])
    v = np.array([sqrt3 / 2, 3 / 2])

    # 7 Core Hexagons (Ring 0 and 1)
    core_indices = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    
    # 5 Side Hexagons from Ring 2
    # Ring 2 Side positions (angles 30, 90, 150, 210, 270)
    # Lattice coords for Side positions:
    # 30 deg: (1, 1)
    # 90 deg: (-1, 2)
    # 150 deg: (-2, 1) -- Wait, (-2,1) is a Corner?
    # Let's re-verify lattice coords for Ring 2.
    # Corners (dist 2sqrt3): (2,0), (0,2), (-2,1), (-2,0), (0,-2), (2,-1)
    # Sides (dist 3): (1,1), (-1,2), (-1,-1), (1,-2), (2,-2), (1,-3)? No.
    # Let's generate them to be safe.
    
    ring2_sides = []
    # Check all lattice points with small i,j
    for i in range(-3, 4):
        for j in range(-3, 4):
            if i == 0 and j == 0: continue
            dist_sq = 3 * (i*i + j*j + i*j)
            # Ring 1: dist_sq = 3
            # Ring 2: dist_sq = 9 (Sides) or 12 (Corners)
            if dist_sq == 9:
                ring2_sides.append((i, j))
                
    # We expect 6 sides. Pick 5.
    # To maintain symmetry as much as possible, we can pick any 5.
    # Let's pick the first 5.
    selected_sides = ring2_sides[:5]
    
    all_indices = core_indices + selected_sides
    
    inner_hex_data = np.zeros((12, 3))
    for k, (i, j) in enumerate(all_indices):
        center = i * u + j * v
        inner_hex_data[k, 0] = center[0]
        inner_hex_data[k, 1] = center[1]
        inner_hex_data[k, 2] = 0  # Angle 0

    outer_hex_side_length = 2 * sqrt3 + 1
    outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
