"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Generate hexagonal lattice points
    # Hexagonal lattice vectors: v1 = (1, 0), v2 = (0.5, sqrt(3)/2)
    # We generate points (i*v1 + j*v2) for i, j in a range
    # Then select the N closest to the origin
    
    # To get 21 points, we need a range large enough.
    # A hexagon of radius 3 has 37 points. Radius 2 has 19.
    # So we generate up to radius 3 and pick the 21 closest.
    
    points = []
    # i, j are integer coordinates in the hex lattice
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2) * j
            points.append((x, y))
            
    points = np.array(points)
    
    # Calculate distances from origin
    dists = np.linalg.norm(points, axis=1)
    
    # Get indices of the 21 closest points
    # Break ties by sorting by angle to ensure a consistent "compact" shape
    # Actually, just sorting by distance is fine, but ties might lead to asymmetry.
    # Let's sort by distance, then by angle.
    angles = np.arctan2(points[:, 1], points[:, 0])
    # Combine distance and angle into a single sort key? 
    # Simpler: sort by distance, then stable sort by angle for ties.
    # Or just take the first 21 from a sorted list of (dist, idx).
    # To ensure reproducibility and symmetry, we can add a small perturbation based on index?
    # No, let's just sort by distance. If there are ties, the order depends on the generation order.
    # To make it more symmetric, we can rotate the lattice or pick points in a symmetric way.
    # For now, let's just sort by distance.
    
    sorted_indices = np.argsort(dists)
    # If there are ties in distance, we might want to break them symmetrically.
    # Let's check the distances of the 21st and 22nd point.
    # If they are equal, we have a choice.
    # For a hexagonal lattice, the 21 points closest to origin form a hexagon of radius 2 (19 pts) + 2 pts from radius 3.
    # The 2 pts from radius 3 should be placed symmetrically to maintain hexagonal symmetry as much as possible.
    # Let's identify the 19 points with dist < 2 (or close to 2).
    # Actually, the distance to the 6th ring point is 2.
    # Points with dist < 2: 
    # Center (0,0) - 1
    # Ring 1 (dist 1) - 6
    # Ring 2 (dist 2) - 12? No, the distance to the 6 "corner" points of ring 2 is 2.
    # The other 6 points in ring 2 have distance sqrt(3) ~ 1.732.
    # So points with dist < 2 are: 1 (center) + 6 (ring 1) + 6 (inner ring 2) = 13.
    # Points with dist = 2 are the 6 corners of ring 2. Total 19.
    # The next ring (ring 3) has points with distance 3 (corners) and sqrt(7) ~ 2.64 (others).
    # So the 20th and 21st points will be from ring 3.
    # To maintain symmetry, we should pick 2 points from ring 3 that are opposite or adjacent in a symmetric way?
    # Actually, adding 2 points to a hexagon of 19 breaks the 6-fold symmetry.
    # Let's just pick the 21 closest. If there's a tie for the 21st spot, we might get an asymmetric shape.
    # Let's force symmetry by picking points based on angle sectors.
    
    # Alternative: Just use the 21 closest, and then optimize the rotation.
    # The rotation will help align the "bulge" of the extra 2 points with the rectangle axes to minimize bounding box.
    
    idx_21 = sorted_indices[:N]
    cluster = points[idx_21]
    
    # Center the cluster
    center = np.mean(cluster, axis=0)
    cluster = cluster - center
    
    # Try different rotations to minimize the bounding box perimeter (width + height)
    best_w_h = float('inf')
    best_rot = 0
    best_cluster = cluster
    
    for angle in np.linspace(0, np.pi/3, 13): # 0 to 60 degrees, 13 points
        c, s = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[c, -s], [s, c]])
        rotated = cluster @ rot_matrix.T
        
        # Bounding box
        min_x, min_y = np.min(rotated, axis=0)
        max_x, max_y = np.max(rotated, axis=0)
        w = max_x - min_x
        h = max_y - min_y
        
        # We want to minimize w + h to maximize r = 1 / (2 + w + h)
        if w + h < best_w_h:
            best_w_h = w + h
            best_rot = angle
            best_cluster = rotated
            best_w = w
            best_h = h
            
    # Calculate optimal radius and scale
    # r = 1 / (2 + w + h)
    r = 1.0 / (2.0 + best_w_h)
    d = 2.0 * r # spacing
    
    # Scale the cluster
    scaled_cluster = best_cluster * d
    
    # The bounding box of the centers is (best_w * d) x (best_h * d)
    # The rectangle must contain the circles, so its size is:
    W = best_w * d + 2 * r
    H = best_h * d + 2 * r
    
    # Check perimeter: 2(W + H) = 2(W + H)
    # W + H = best_w * d + 2r + best_h * d + 2r = (best_w + best_h) * d + 4r
    # d = 2r
    # W + H = (best_w + best_h) * 2r + 4r = 2r (best_w + best_h + 2)
    # Perimeter = 2(W+H) = 4r (best_w + best_h + 2)
    # We set r = 1 / (2 + best_w_h), so Perimeter = 4 * (1 / (2 + best_w_h)) * (2 + best_w_h) = 4.
    # Correct.
    
    # Center the scaled cluster in the rectangle [0, W] x [0, H]
    # The center of the cluster is at (0,0) in scaled_cluster coordinates.
    # We want to place it at (W/2, H/2).
    # But we also need to ensure the circles don't cross the boundary.
    # The min x of scaled_cluster is -best_w/2 * d.
    # The left edge of the leftmost circle is -best_w/2 * d - r.
    # We want this to be >= 0.
    # Let's shift the cluster so that the leftmost circle touches x=0 and bottommost touches y=0.
    
    min_x_c, min_y_c = np.min(scaled_cluster, axis=0)
    max_x_c, max_y_c = np.max(scaled_cluster, axis=0)
    
    # Shift so that min_x_c - r = 0 => min_x_c = r
    # And min_y_c - r = 0 => min_y_c = r
    shift_x = r - min_x_c
    shift_y = r - min_y_c
    
    final_centers = scaled_cluster + np.array([shift_x, shift_y])
    
    # Construct the output array
    result = np.zeros((N, 3))
    result[:, 0] = final_centers[:, 0]
    result[:, 1] = final_centers[:, 1]
    result[:, 2] = r
    
    return result
# EVOLVE-BLOCK-END
