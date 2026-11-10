import numpy as np
import time

def hexagon_packing_12():
    start_time = time.time()
    
    # Axial coordinates for 12 hexagons
    # We select 12 cells from the 13 cells with q^2 + qr + r^2 <= 2
    # These cells have center distance <= sqrt(6) from origin.
    # Vertices are at distance <= sqrt(6) + 1.
    # sqrt(6) + 1 ≈ 3.449
    # sqrt(12) ≈ 3.464
    # So they fit in a hexagon of side sqrt(12).
    
    # The 13 cells are:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (1,1), (-1,-1), (2,-1), (-2,1), (1,-2), (-1,2)
    
    # We remove (1,1) to get 12 cells.
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (-1, -1), (2, -1), (-2, 1), (1, -2), (-1, 2)
    ]
    
    # Convert to Cartesian
    # x = 1.5 * q
    # y = sqrt(3) * (r + q / 2.0)
    points = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        points.append([x, y])
    points = np.array(points)
    
    # Center the cluster
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # The optimal side length s is constrained by s >= sqrt(12)
    # And the geometry fits in s = sqrt(6) + 1 < sqrt(12)
    # So we use s = sqrt(12) to maximize the score 1/s (assuming proportionality)
    # Wait, the score is 1/s. Smaller s is better.
    # So we want the SMALLEST valid s.
    # The smallest valid s is max(sqrt(12), geometric_min_s).
    # geometric_min_s = sqrt(6) + 1.
    # Since sqrt(12) > sqrt(6) + 1, the smallest valid s is sqrt(12).
    
    s_opt = np.sqrt(12)
    
    # Construct inner hexagon data
    # Angle 0 for all inner hexagons
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = points[i, 0]
        inner_hex_data[i, 1] = points[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    # Outer hexagon centered at 0,0, angle 0
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_opt
