import math
import numpy as np
import time

# Constants
APOTHEM = math.sqrt(3) / 2.0

def construct():
    """
    Constructs a packing of 11 unit regular hexagons inside a regular hexagon.
    Uses a lattice packing within a hexagon of side length 3.
    """
    # Generate lattice points for a hexagon of side 3 (radius 2 in axial coords)
    # Points (q, r) such that max(|q|, |r|, |q+r|) <= 2
    points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q + r)) <= 2:
                points.append((q, r))
    
    # We need 11 points. The list 'points' contains 19 points.
    # We select a compact subset: center, 6 neighbors, and 4 from the next ring.
    # Center
    selected = [(0, 0)]
    # 6 Neighbors
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    selected.extend(neighbors)
    # 4 from the next ring (Ring 2)
    # We pick (2,0), (1,1), (0,2), (-1,2) to maintain symmetry/compactness
    # Note: The full ring 2 has 12 points. We just need 4 more.
    ring2_subset = [(2, 0), (1, 1), (0, 2), (-1, 2)]
    selected.extend(ring2_subset)
    
    # Ensure we have exactly 11
    if len(selected) > 11:
        selected = selected[:11]
    elif len(selected) < 11:
        # Fallback: add more from the full list if needed (should not happen)
        for p in points:
            if p not in selected:
                selected.append(p)
            if len(selected) == 11:
                break

    # Convert axial coordinates to Cartesian
    inner = []
    for q, r in selected:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([float(x), float(y), 0.0])

    # Container definition
    # Side length 3.0
    # Center (0,0)
    # Rotation 0.0
    center = [0.0, 0.0]
    side = 3.0
    rotation = 0.0

    return inner, center, side, rotation
