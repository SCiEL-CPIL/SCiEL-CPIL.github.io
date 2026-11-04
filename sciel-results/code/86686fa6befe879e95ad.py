import numpy as np
from itertools import combinations

N = 11

def construct():
    # Centers of the 7-hexagon cluster (Pointy-Top)
    # Center
    c0 = (0.0, 0.0)
    # Neighbors
    d = np.sqrt(3)
    neighbors = [
        (d, 0.0),
        (d/2, 1.5),
        (-d/2, 1.5),
        (-d, 0.0),
        (-d/2, -1.5),
        (d/2, -1.5)
    ]
    
    base_centers = [c0] + neighbors
    
    # Vertices of base centers
    # Pointy-top unit hex vertices relative to center
    # (0, 1), (sqrt(3)/2, 0.5), (sqrt(3)/2, -0.5), (0, -1), (-sqrt(3)/2, -0.5), (-sqrt(3)/2, 0.5)
    rel_verts = np.array([
        [0.0, 1.0],
        [np.sqrt(3)/2, 0.5],
        [np.sqrt(3)/2, -0.5],
        [0.0, -1.0],
        [-np.sqrt(3)/2, -0.5],
        [-np.sqrt(3)/2, 0.5]
    ])
    
    base_verts = np.vstack([v + np.array(c) for c, v in zip(base_centers, [rel_verts]*7)])
    
    # Attachment centers (2 * neighbor)
    attach_centers = [2 * np.array(n) for n in neighbors]
    
    best_S = float('inf')
    best_theta_rad = 0.0
    best_sides = None
    best_config_centers = None
    
    # Combinations of 4 sides
    for sides in combinations(range(6), 4):
        # Add vertices for attached hexagons
        extra_verts_list = []
        for s in sides:
            c = attach_centers[s]
            extra_verts_list.append(rel_verts + c)
        
        if extra_verts_list:
            extra_verts = np.vstack(extra_verts_list)
            all_verts = np.vstack([base_verts, extra_verts])
        else:
            all_verts = base_verts
            
        # Optimize theta
        # S(theta) is max of projections
        # We can sample theta
        thetas = np.linspace(0, np.pi/3, 200)
        for theta in thetas:
            c_t, s_t = np.cos(theta), np.sin(theta)
            R = np.array([[c_t, -s_t], [s_t, c_t]])
            verts_rot = all_verts @ R.T
            
            x = verts_rot[:, 0]
            y = verts_rot[:, 1]
            
            S1 = np.max(np.abs(y))
            S2 = np.max(np.abs(x + np.sqrt(3)*y)) / np.sqrt(3)
            S3 = np.max(np.abs(x - np.sqrt(3)*y)) / np.sqrt(3)
            
            S = max(S1, S2, S3)
            
            if S < best_S:
                best_S = S
                best_theta_rad = theta
                best_sides = sides
                best_config_centers = base_centers + [tuple(attach_centers[s]) for s in sides]

    if best_config_centers is None:
        return [[0.0, 0.0, 0.0]], (0.0, 0.0), 40.0, 0.0

    # Convert theta to degrees
    outer_rotation_deg = np.rad2deg(-best_theta_rad)
    
    inner = [[c[0], c[1], 0.0] for c in best_config_centers]
    
    return inner, (0.0, 0.0), best_S, outer_rotation_deg
