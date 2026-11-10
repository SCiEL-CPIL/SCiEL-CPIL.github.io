import numpy as np

def hexagon_packing_11():
    # 1. Define centers for 11 hexagons
    sqrt3 = np.sqrt(3)
    
    # Center
    centers = [(0, 0)]
    
    # Ring 1: 6 hexagons at distance sqrt(3)
    for i in range(6):
        angle = i * np.pi / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        centers.append((x, y))
        
    # Ring 2 (Inner): 4 hexagons at distance 3
    # Angles: 30, 150, 210, 330 degrees
    angles_r2 = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for angle in angles_r2:
        x = 3 * np.cos(angle)
        y = 3 * np.sin(angle)
        centers.append((x, y))
        
    # Scale centers to ensure separation > 1e-6
    k = 1.0 + 1e-5
    centers = np.array(centers) * k
    
    # 2. Generate vertices for all inner hexagons
    hex_verts_rel = np.array([
        [1, 0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1, 0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    all_vertices = []
    for cx, cy in centers:
        verts = hex_verts_rel + np.array([cx, cy])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # 3. Optimize outer hexagon orientation
    # We want to minimize the apothem A required to contain all_vertices.
    # A = max_p max_k | p dot n_k |
    # n_k are normals at angles theta, theta+60, theta+120.
    
    best_A = float('inf')
    best_theta = 0
    
    thetas = np.linspace(0, np.pi/6, 2000)
    
    for theta in thetas:
        n_angles = [theta, theta + np.pi/3, theta + 2*np.pi/3]
        normals = np.array([[np.cos(a), np.sin(a)] for a in n_angles])
        
        proj = all_vertices @ normals.T
        max_proj = np.max(np.abs(proj))
        
        if max_proj < best_A:
            best_A = max_proj
            best_theta = theta
            
    S = 2 * best_A / sqrt3
    
    # 4. Prepare output
    inner_hex_data = np.zeros((11, 3))
    for i, (cx, cy) in enumerate(centers):
        inner_hex_data[i, 0] = cx
        inner_hex_data[i, 1] = cy
        inner_hex_data[i, 2] = 0.0
        
    # Outer hexagon vertex angle 'a' in formula cx + s*cos(a + 60i)
    # Normals are at a + 30 + 60i.
    # We want normals at best_theta + 60i.
    # So a + 30 = best_theta => a = best_theta - 30.
    outer_angle_deg = np.degrees(best_theta - np.pi/6)
    
    outer_hex_data = np.array([0.0, 0.0, outer_angle_deg])
    
    return inner_hex_data, outer_hex_data, S
