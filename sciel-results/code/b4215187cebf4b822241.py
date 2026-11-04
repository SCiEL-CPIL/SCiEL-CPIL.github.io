import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the grid parameters
    # 3 rows, 4 columns of staggered hexagons
    rows = 3
    cols = 4
    
    # Lattice vectors for pointy-top hexagons (angle 0)
    # dx = sqrt(3), dy = 1.5 for the staggered step
    sqrt3 = np.sqrt(3.0)
    
    centers = []
    for i in range(rows):
        for j in range(cols):
            # x offset: j * sqrt(3) + i * sqrt(3)/2
            # y offset: i * 1.5
            x = j * sqrt3 + (i % 2) * (sqrt3 / 2.0)
            y = i * 1.5
            centers.append([x, y, 0.0]) # angle 0.0
    
    # Center the cluster at (0,0)
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    
    for c in centers:
        c[0] -= cx
        c[1] -= cy
        
    # Calculate the required outer side length S
    # Outer hexagon is also angle 0.0, centered at (0,0)
    # Constraints for containment in hexagon with apothem A = S * sqrt(3) / 2:
    # 1. |x| <= A
    # 2. |0.5*x + sqrt(3)/2*y| <= A
    # 3. |-0.5*x + sqrt(3)/2*y| <= A
    
    max_abs_x = max(abs(c[0]) for c in centers)
    # The hexagon itself has radius 1 (circumradius) in x direction for angle 0
    # So effective max x extent is max_abs_x + 1
    eff_max_x = max_abs_x + 1.0
    
    # For the other constraints, we need to check the corners of the bounding box
    # Or simply iterate over all points and check the max value of the linear forms
    # Since the hexagon is convex and the condition is linear, the max is at the extreme points of the set.
    # The set is the union of hexagons. The extreme points are the vertices of the hexagons.
    # However, for a safe lower bound, we can use the bounding box of the centers + radius.
    # A tighter bound:
    # Max value of 0.5*x + sqrt(3)/2*y over all points in the hexagons.
    # This is equivalent to max(0.5*x_c + sqrt(3)/2*y_c) + max(0.5*vx + sqrt(3)/2*vy) for v in hexagon vertices.
    # For a unit hexagon at angle 0, vertices are (1,0), (0.5, sqrt3/2), (-0.5, sqrt3/2), etc.
    # Let's just compute the max over all vertices of all inner hexagons.
    
    max_lin1 = 0.0
    max_lin2 = 0.0
    
    # Vertices of a unit hexagon at angle 0 relative to center
    verts = [
        (1.0, 0.0),
        (0.5, sqrt3/2.0),
        (-0.5, sqrt3/2.0),
        (-1.0, 0.0),
        (-0.5, -sqrt3/2.0),
        (0.5, -sqrt3/2.0)
    ]
    
    for c in centers:
        xc, yc = c[0], c[1]
        for vx, vy in verts:
            px = xc + vx
            py = yc + vy
            val1 = 0.5 * px + (sqrt3 / 2.0) * py
            val2 = -0.5 * px + (sqrt3 / 2.0) * py
            if abs(val1) > abs(max_lin1):
                max_lin1 = val1
            if abs(val2) > abs(max_lin2):
                max_lin2 = val2
            if abs(px) > abs(max_abs_x): # Update max_abs_x with vertices just in case
                max_abs_x = px

    # We need A >= max(max_abs_x, abs(max_lin1), abs(max_lin2))
    # A = S * sqrt(3) / 2
    # S = 2 * A / sqrt(3)
    
    required_apothem = max(abs(max_abs_x), abs(max_lin1), abs(max_lin2))
    # Add a tiny epsilon to ensure strict inequality or numerical stability
    required_apothem += 1e-6 
    
    S = (2.0 * required_apothem) / sqrt3
    
    return centers, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
