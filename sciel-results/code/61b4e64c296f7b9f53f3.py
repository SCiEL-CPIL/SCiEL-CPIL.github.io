import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 hexagon centers
    # 7 core
    centers = [
        (0.0, 0.0),
        (np.sqrt(3), 0.0),
        (np.sqrt(3)/2, 1.5),
        (-np.sqrt(3)/2, 1.5),
        (-np.sqrt(3), 0.0),
        (-np.sqrt(3)/2, -1.5),
        (np.sqrt(3)/2, -1.5)
    ]
    # 4 edge-midpoints from the next layer
    # (2,-1), (1,-2), (-1,-1), (-2,1) in axial coords
    # Convert to Cartesian
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        return x, y

    centers.append(axial_to_cartesian(2, -1))
    centers.append(axial_to_cartesian(1, -2))
    centers.append(axial_to_cartesian(-1, -1))
    centers.append(axial_to_cartesian(-2, 1))
    
    # All rotations are 0
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Calculate required side length L
    # Vertices of a unit hexagon at (cx, cy) with rot 0:
    # (cx+1, cy), (cx+0.5, cy+sqrt(3)/2), (cx-0.5, cy+sqrt(3)/2),
    # (cx-1, cy), (cx-0.5, cy-sqrt(3)/2), (cx+0.5, cy-sqrt(3)/2)
    
    max_L = 0.0
    for cx, cy, _ in inner:
        verts = [
            (cx + 1.0, cy),
            (cx + 0.5, cy + np.sqrt(3)/2.0),
            (cx - 0.5, cy + np.sqrt(3)/2.0),
            (cx - 1.0, cy),
            (cx - 0.5, cy - np.sqrt(3)/2.0),
            (cx + 0.5, cy - np.sqrt(3)/2.0)
        ]
        for vx, vy in verts:
            # Constraints for outer hexagon side L, rot 0
            # |x| <= L
            # |y| <= L * sqrt(3) / 2
            # |sqrt(3)*x + y| <= L * sqrt(3)
            # |sqrt(3)*x - y| <= L * sqrt(3)
            
            L1 = abs(vx)
            L2 = abs(vy) * 2.0 / np.sqrt(3)
            L3 = abs(np.sqrt(3)*vx + vy) / np.sqrt(3)
            L4 = abs(np.sqrt(3)*vx - vy) / np.sqrt(3)
            
            current_L = max(L1, L2, L3, L4)
            if current_L > max_L:
                max_L = current_L
                
    # Add a tiny epsilon for safety
    side_length = max_L + 1e-9
    
    return inner, (0.0, 0.0), side_length, 0.0
# EVOLVE-BLOCK-END
