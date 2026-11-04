import numpy as np

def construct():
    # Direct construction of the known optimal 'flower' packing for 11 hexagons.
    # This configuration fits exactly in a regular hexagon of side length 4.
    # We apply a small scaling factor to the centers to ensure strict non-overlap
    # and containment, avoiding floating-point precision issues.
    
    # Triangular lattice vectors for tangent hexagons
    u = np.array([1.5, np.sqrt(3)/2])
    v = np.array([0.0, np.sqrt(3)])
    
    # Lattice coordinates for the 11 hexagons
    # 1 center, 6 neighbors, 4 specific second-ring positions
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (-2, 1), (1, 1), (-1, -1)
    ]
    
    # Scaling factor to ensure non-overlap
    # Original max center distance is 3.0.
    # Original min center distance is sqrt(3).
    # k=1.001 provides a small but robust gap.
    k = 1.001
    
    # Calculate centers
    centers = []
    for i, j in lattice_coords:
        center = (i * u + j * v) * k
        centers.append(center)
    
    # Inner hexagons: (x, y, angle_degrees)
    # All angles are 0 degrees (aligned with the outer hexagon's orientation)
    inner = []
    for c in centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
    
    # Outer hexagon parameters
    # Center is at the origin
    outer_center = (0.0, 0.0)
    
    # Calculate required outer side length
    # The farthest centers are at distance 3*k from the origin.
    # The inner hexagons have radius 1.
    # So the max extent is 3*k + 1.
    # The outer hexagon (angle 0) has vertices at distance S from center.
    # So S must be >= 3*k + 1.
    max_extent = 3.0 * k + 1.0
    
    # Add a small epsilon for safety margin in containment check
    outer_side = max_extent + 1e-6
    
    # Angle is 0 degrees
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
