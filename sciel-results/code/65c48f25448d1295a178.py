import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Define the 11 hexagon centers in a compact cluster
    # Using axial coordinates converted to Cartesian
    # v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 1.5)
    sqrt3 = np.sqrt(3)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2, 1.5])
    
    # 7 central hexagons (radius 1 in lattice steps)
    # (0,0) and its 6 neighbors
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1)
    ]
    
    # Add 4 more hexagons to make 11
    # To keep it compact, we add them in a way that extends the shape slightly
    # Let's try adding (2,0), (-2,0), (1,1), (-1,-1) ? 
    # Or maybe (2,1), (-2,-1), (1,2), (-1,-2)
    
    # Let's try a configuration that is roughly circular
    # Add (2, 0), (-2, 0), (1, 1), (-1, -1)
    # Note: (1,1) is not a neighbor of (0,0) but is in the lattice.
    # Distance (1,1) to (0,0) is sqrt(3^2 + 1.5^2)? No.
    # (1,1) = v1 + v2 = (1.5sqrt3, 1.5). Dist = 3.
    
    # Let's pick 4 points from the second shell (dist 3)
    # (2,0), (-2,0), (1,1), (-1,-1)
    # (2,0) = 2*v1 = (2sqrt3, 0)
    # (-2,0) = (-2sqrt3, 0)
    # (1,1) = v1+v2 = (1.5sqrt3, 1.5)
    # (-1,-1) = -v1-v2 = (-1.5sqrt3, -1.5)
    
    axial_coords.extend([
        (2, 0),
        (-2, 0),
        (1, 1),
        (-1, -1)
    ])
    
    # Convert to Cartesian
    centers = []
    for q, r in axial_coords:
        x = q * v1[0] + r * v2[0]
        y = q * v1[1] + r * v2[1]
        centers.append([x, y])
    
    centers = np.array(centers)
    
    # Function to check if a point is inside a regular hexagon
    # Outer hexagon: center (0,0), side L, angle theta
    # Vertices of outer hexagon are at radius L.
    # Apothem a = L * sqrt(3) / 2
    # The hexagon is the intersection of 6 half-planes.
    # Normal vectors are at angles theta + k*60 deg.
    
    def min_side_length(centers, theta):
        # For a fixed theta, find the minimum L such that all vertices of all inner hexagons are inside.
        # This is equivalent to: L >= max over all vertices v of (support function of v in direction n_i) / (sqrt(3)/2)
        # Actually, if the hexagon is centered at origin, and has apothem a, then for any point p,
        # p is inside if dot(p, n_i) <= a for all 6 normals n_i.
        # So a >= max_i max_p dot(p, n_i).
        # L = 2 * a / sqrt(3).
        
        normals = []
        for k in range(6):
            angle = theta + k * np.pi / 3
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_proj = 0.0
        # Get all vertices of all inner hexagons
        # Inner hexagons have side 1.
        # Vertices of a unit hexagon centered at C with angle 0:
        # (1, 0), (0.5, sqrt(3)/2), ...
        
        # Optimization: We can just check the vertices of the inner hexagons.
        # There are 11 * 6 = 66 vertices.
        
        # Precompute local vertices for a unit hexagon
        local_verts = np.array([
            [1.0, 0.0],
            [0.5, sqrt3/2],
            [-0.5, sqrt3/2],
            [-1.0, 0.0],
            [-0.5, -sqrt3/2],
            [0.5, -sqrt3/2]
        ])
        
        for c in centers:
            global_verts = c + local_verts
            for n in normals:
                projs = np.dot(global_verts, n)
                max_proj = max(max_proj, np.max(projs))
        
        a = max_proj
        L = 2 * a / sqrt3
        return L

    # Search for optimal theta
    best_L = float('inf')
    best_theta = 0.0
    
    # Sample theta from 0 to 60 degrees (symmetry)
    num_samples = 100
    for i in range(num_samples):
        theta = i * np.pi / 3 / num_samples
        L = min_side_length(centers, theta)
        if L < best_L:
            best_L = L
            best_theta = theta
        
        if time.time() > deadline:
            break
    
    # Refine around best_theta
    if time.time() < deadline - 5:
        step = np.pi / 3 / 1000
        for i in range(-10, 11):
            theta = best_theta + i * step
            L = min_side_length(centers, theta)
            if L < best_L:
                best_L = L
                best_theta = theta

    # Construct return values
    # inner: list of [x, y, angle_degrees]
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0]) # Inner hexagons can be rotated, but 0 is fine if we accounted for it in vertex calc?
        # Wait, my vertex calculation assumed inner hexagons are at angle 0.
        # If we allow inner hexagons to rotate, we might fit them tighter?
        # Usually, for hexagons, aligning them is optimal.
        
    return inner, (0.0, 0.0), best_L, best_theta * 180 / np.pi
# EVOLVE-BLOCK-END
