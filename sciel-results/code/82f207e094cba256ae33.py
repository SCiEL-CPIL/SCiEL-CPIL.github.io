# EVOLVE-BLOCK-START
import numpy as np

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a hexagonal lattice ring configuration (12 hexes forming a ring around the center).
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_degrees).
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_degrees).
        outer_hex_side_length: float.
    """
    # Generate the 12 centers for the outer ring of a side-2 hexagon cluster
    # Lattice basis
    e1 = np.array([np.sqrt(3), 0])
    e2 = np.array([np.sqrt(3)/2, 1.5])
    
    centers = []
    # The 12 centers correspond to axial coordinates (q, r) where max(|q|, |r|, |q+r|) == 2
    # and not all 0.
    # We iterate through q, r in range -2 to 2
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) == 2:
                # Convert axial to Cartesian
                x = np.sqrt(3) * (q + r/2)
                y = 1.5 * r
                centers.append([x, y, 0.0])
                
    # We should have exactly 12 centers
    inner_hex_data = np.array(centers)
    
    # Calculate optimal side length s
    # s = 2 * sqrt(3) + 1
    s = 2 * np.sqrt(3) + 1
    
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
