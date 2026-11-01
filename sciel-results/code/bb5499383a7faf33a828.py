import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate 12 unit hexagons in a 4x3 rhombus arrangement
    # Using pointy-top orientation (angle 0) for inner hexagons
    # Lattice vectors for pointy-top:
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    
    sqrt3 = np.sqrt(3)
    centers = []
    # 4 columns (i), 3 rows (j)
    # i in 0..3, j in 0..2
    for i in range(4):
        for j in range(3):
            x = sqrt3 * (i + j / 2.0)
            y = 1.5 * j
            centers.append([x, y, 0.0])
            
    # Center the cluster
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    inner = [[c[0] - cx, c[1] - cy, c[2]] for c in centers]
    
    # Outer hexagon
    # We determined S=5.0 with 30 degree rotation works well
    # The cluster is symmetric, so center is (0,0)
    # Rotation of outer hexagon: 30 degrees (pointy-top)
    # Side length: 5.0
    side = 5.0
    angle = 30.0
    
    return inner, (0.0, 0.0), side, angle
# EVOLVE-BLOCK-END
