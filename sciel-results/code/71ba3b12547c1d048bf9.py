import numpy as np

def construct():
    N = 11
    
    # Lattice vectors for hexagonal packing of unit hexagons (flat-top)
    # Distance between centers is sqrt(3)
    v1 = np.array([np.sqrt(3), 0.0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate candidate lattice points
    # We need 11 points. The 11 closest to origin are sufficient.
    # Let's generate a grid of lattice indices (i, j)
    points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            p = i * v1 + j * v2
            points.append(p)
            
    points = np.array(points)
    
    # Calculate distance from origin
    dists = np.linalg.norm(points, axis=1)
    
    # Get indices of 11 closest points
    # Note: There are ties in distance. We just take the first 11 found in the sorted order.
    # To be robust, we can sort by distance, then by index.
    indices = np.argsort(dists)
    selected_indices = indices[:N]
    
    centers = points[selected_indices]
    
    # The centers are relative to (0,0).
    # We can optionally center them by subtracting the mean to minimize bounding box?
    # For the symmetric set of 11 (1+6+4), if we pick symmetric 4s, mean is 0.
    # If we pick arbitrary 4s, mean might not be 0.
    # Let's center them to be safe.
    center_offset = np.mean(centers, axis=0)
    centers = centers - center_offset
    
    # Construct inner list
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    # Calculate required side length S
    # Normals for flat-top hexagon
    normals = []
    for k in range(6):
        angle = k * np.pi / 3
        normals.append([np.cos(angle), np.sin(angle)])
    normals = np.array(normals)
    
    # Max projection of centers
    # proj[i, k] = center_i dot normal_k
    projs = np.dot(centers, normals.T)
    max_proj = np.max(projs)
    
    # S * sqrt(3)/2 >= max_proj + sqrt(3)/2
    S = (max_proj + np.sqrt(3)/2) * 2 / np.sqrt(3)
    
    # Add a tiny epsilon to ensure validity
    S += 1e-6
    
    outer_center = [0.0, 0.0]
    outer_side = S
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
