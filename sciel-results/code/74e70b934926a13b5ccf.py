import numpy as np
from scipy.optimize import minimize

def hexagon_packing_12():
    # Use a hexagonal lattice and find the best 12-point cluster
    # that minimizes the bounding regular hexagon side length
    
    # Lattice vectors for honeycomb packing
    # Neighbor distance is sqrt(3)
    u1 = np.array([1.5, np.sqrt(3) / 2.0])
    u2 = np.array([-1.5, np.sqrt(3) / 2.0])
    
    # Generate a large grid of candidate lattice points
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            c = i * u1 + j * u2
            points.append(c)
    points = np.array(points)
    
    # Center the grid
    points -= points.mean(axis=0)
    
    # We need to select 12 points from this grid that minimize the bounding hexagon
    # Use a greedy + local search approach
    
    # First, try several candidate arrangements
    # Arrangement 1: 3x4 rectangular grid (current)
    centers_rect = []
    for i in range(3):
        for j in range(4):
            c = i * u1 + j * u2
            centers_rect.append(c)
    centers_rect = np.array(centers_rect)
    centers_rect -= centers_rect.mean(axis=0)
    
    # Arrangement 2: Hexagonal cluster - center + 6 neighbors + 5 more
    # Center at origin
    # 6 neighbors at distance sqrt(3)
    # Then 5 more from the second ring
    
    # Let me define specific hexagonal cluster arrangements
    
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
    s_rect = compute_s(centers_rect)
    
    # Try hexagonal cluster: use lattice points near center
    # Select 12 points from the lattice that are closest to center
    # This should give a compact hexagonal shape
    
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    
    # Take the 12 closest points to center
    centers_hex = points[sorted_idx[:12]]
    s_hex = compute_s(centers_hex)
    
    # Choose the better arrangement
    if s_hex < s_rect:
        centers = centers_hex
        s = s_hex
    else:
        centers = centers_rect
        s = s_rect
    
    # Scale to ensure separation > 1e-6
    scale = 1.0 + 1e-6
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
