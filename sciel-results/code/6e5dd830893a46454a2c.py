def point_in_hexagon(p, center, side, angle_deg):
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    angle = np.radians(angle_deg)
    # Normals are at angle + 30 + k*60
    # k=0..5
    # max_proj = max(dx * cos(phi_k) + dy * sin(phi_k))
    # If max_proj <= side * sqrt(3)/2, inside.
    
    sqrt3_over_2 = np.sqrt(3) / 2.0
    limit = side * sqrt3_over_2
    
    max_proj = -np.inf
    for k in range(6):
        phi = angle + np.pi/6 + k * np.pi/3
        proj = dx * np.cos(phi) + dy * np.sin(phi)
        if proj > max_proj:
            max_proj = proj
            
    return max_proj <= limit + 1e-9, max_proj - limit
