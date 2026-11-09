import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define 11 hexagon centers in a compact cluster
    # Base 7 hexagons (center + 6 neighbors)
    # Spacing: dx = sqrt(3), dy = 1.5
    s3 = np.sqrt(3)
    
    centers = [
        (0.0, 0.0),
        (s3, 0.0),
        (s3/2, 1.5),
        (-s3/2, 1.5),
        (-s3, 0.0),
        (-s3/2, -1.5),
        (s3/2, -1.5)
    ]
    
    # Add 4 more to make 11
    # Placing them on the sides to maintain compactness
    centers.append((3*s3/2, 1.5))
    centers.append((3*s3/2, -1.5))
    centers.append((-3*s3/2, 1.5))
    centers.append((-3*s3/2, -1.5))
    
    # Ensure we have exactly 11
    centers = centers[:N]
    
    # Convert to list of [x, y, angle]
    # All inner hexagons are aligned (angle 0)
    inner = [[c[0], c[1], 0.0] for c in centers]
    
    center = (0.0, 0.0)
    
    # Function to calculate max projection of the cluster in a given direction
    def max_projection(direction_angle):
        ca = np.cos(direction_angle)
        sa = np.sin(direction_angle)
        max_proj = -np.inf
        for x, y, _ in inner:
            # Projection of center
            proj = x * ca + y * sa
            # Support function of unit hexagon (side 1, rot 0) in this direction
            # h(phi) = (sqrt(3)/2) * sec( (phi mod pi/3) - pi/6 )
            # phi is direction angle
            # We need h(direction_angle - 0)
            delta = (direction_angle % (np.pi/3)) - np.pi/6
            h = (np.sqrt(3)/2) / np.cos(delta)
            max_proj = max(max_proj, proj + h)
        return max_proj

    best_L = np.inf
    best_rot = 0.0
    
    # Try rotating the outer hexagon
    # The outer hexagon's constraints are defined by its 6 side normals.
    # If outer rot is theta, normals are theta, theta+60, ...
    # We need L such that for all k, max_proj(theta + k*pi/3) <= L * sqrt(3) / 2
    # So L = max_k ( max_proj(theta + k*pi/3) * 2 / sqrt(3) )
    
    for i in range(60):
        theta = i * np.pi / 180.0
        max_ext = 0.0
        for k in range(6):
            phi = theta + k * np.pi / 3.0
            ext = max_projection(phi)
            if ext > max_ext:
                max_ext = ext
        L = max_ext * 2.0 / np.sqrt(3)
        if L < best_L:
            best_L = L
            best_rot = np.degrees(theta)

    return inner, center, best_L, best_rot
# EVOLVE-BLOCK-END
