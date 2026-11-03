import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Lattice basis vectors for triangular lattice (hexagonal tiling)
    # Unit hexagons have side length 1.
    # Distance between centers is sqrt(3).
    # We define the lattice in Cartesian coordinates.
    # Let the hexagons be "pointy-top" or "flat-top"?
    # The problem doesn't specify, but we can return the angle.
    # Standard orientation: Flat top (vertices at 0, 60, 120...).
    # Or Pointy top (vertices at 30, 90...).
    # Let's use Pointy Top for easier lattice alignment with standard axes?
    # Actually, it doesn't matter as long as we are consistent.
    # Let's use the lattice where neighbors are at 0, 60, 120, 180, 240, 300 degrees.
    
    # We will construct a cluster of 12 hexagons on this lattice.
    # The most compact cluster of 12 hexagons is a subset of the "radius 2" hexagon (19 hexes).
    # Specifically, the "radius 1" hexagon has 7 hexes.
    # Adding the 6 corner hexes of the radius 2 hexagon gives 13 hexes.
    # We need 12, so we remove 1 hex from the 13-hex cluster.
    
    # Lattice coordinates (axial) to Cartesian (pointy top, center at 0,0)
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # q, r are axial coordinates.
    
    # Generate the 13-hex cluster (center + 6 neighbors + 6 corners)
    # Center: (0,0)
    # Neighbors (dist 1): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Corners (dist 2): (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    
    hexes = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (-2, 0), (0, -2), (2, -2)
    ]
    
    # We have 13 hexes. Remove one to get 12.
    # Removing any one should result in a bounding hexagon of side 3.
    # Let's remove (2,0) for determinism.
    hexes = hexes[:12] # Removes the last one (2,-2)? No, list order.
    # Let's explicitly pick 12.
    selected = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (-2, 0), (0, -2)
    ]
    
    # Convert to Cartesian
    inner = []
    for q, r in selected:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0]) # Angle 0
        
    # The bounding hexagon side length
    # For this cluster, the max width is 3 * sqrt(3).
    # Side length S = Width / sqrt(3) = 3.
    side = 3.0
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
