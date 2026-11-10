import numpy as np
import time

def hexagon_packing_12():
    # Use a hexagonal lattice and find the best 12-point cluster
    # that minimizes the bounding regular hexagon side length
    
    # Lattice vectors for honeycomb packing
    # Neighbor distance is sqrt(3)
    u1 = np.array([1.5, np.sqrt(3) / 2.0])
    u2 = np.array([-1.5, np.sqrt(3) / 2.0])
    
    # Generate a large grid of candidate lattice points
    points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            c = i * u1 + j * u2
            points.append(c)
    points = np.array(points)
    
    # Center the grid
    points -= points.mean(axis=0)
    
    def compute_s(centers):
        """Compute the minimum side length s for outer hexagon"""
        normal_angles = np.pi/6 + np.arange(6) * np.pi/3
        normals = np.array([[np.cos(a), np.sin(a)] for a in normal_angles])
        
        vertex_angles = np.arange(6) * np.pi/3
        unit_vertices = np.array([[np.cos(a), np.sin(a)] for a in vertex_angles])
        
        max_proj = 0.0
        for c in centers:
            for u in unit_vertices:
                v = c + u
                for n in normals:
                    proj = np.dot(v, n)
                    if proj > max_proj:
                        max_proj = proj
        
        return 2.0 * max_proj / np.sqrt(3)
    
    # Try the rectangular arrangement
    centers_rect = []
    for i in range(3):
        for j in range(4):
            c = i * u1 + j * u2
            centers_rect.append(c)
    centers_rect = np.array(centers_rect)
    centers_rect -= centers_rect.mean(axis=0)
    s_rect = compute_s(centers_rect)
    
    # Try hexagonal cluster: use lattice points near center
    # Select 12 points from the lattice that are closest to center
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    centers_hex = points[sorted_idx[:12]]
    s_hex = compute_s(centers_hex)
    
    # Try a 2x6 arrangement (two rows of 6)
    centers_2x6 = []
    for i in range(2):
        for j in range(6):
            c = i * u1 + j * u2
            centers_2x6.append(c)
    centers_2x6 = np.array(centers_2x6)
    centers_2x6 -= centers_2x6.mean(axis=0)
    s_2x6 = compute_s(centers_2x6)
    
    # Try a 4x3 arrangement (four rows of 3)
    centers_4x3 = []
    for i in range(4):
        for j in range(3):
            c = i * u1 + j * u2
            centers_4x3.append(c)
    centers_4x3 = np.array(centers_4x3)
    centers_4x3 -= centers_4x3.mean(axis=0)
    s_4x3 = compute_s(centers_4x3)
    
    # Choose the best arrangement
    best_s = min(s_rect, s_hex, s_2x6, s_4x3)
    if best_s == s_rect:
        centers = centers_rect
    elif best_s == s_hex:
        centers = centers_hex
    elif best_s == s_2x6:
        centers = centers_2x6
    else:
        centers = centers_4x3
    s = best_s
    
    # Use a smaller safety margin (measured improvement)
    scale = 1.0 + 5.8e-7
    centers = centers * scale
    s = s * scale
    
    # Construct output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s
