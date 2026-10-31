def distance_to_hex_boundary(p, cx, cy, angle, size=1.0):
    dx = p[0] - cx
    dy = p[1] - cy
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    normal_angles = np.pi/6 + np.arange(6) * np.pi/3
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    dists = np.dot(np.array([lx, ly]), normals.T)
    apothem = size * np.sqrt(3)/2
    
    # Max of (dists - apothem)
    # If all dists < apothem, max is negative.
    # If any dist > apothem, max is positive.
    return np.max(dists - apothem)
