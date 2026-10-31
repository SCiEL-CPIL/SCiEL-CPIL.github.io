def hexagons_overlap(pos1, ang1, pos2, ang2, eps=1e-7):
    """
    Check if two unit regular hexagons overlap using Separating Axis Theorem.
    pos1, pos2: (x, y) tuples or arrays
    ang1, ang2: angles in radians
    """
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    
    # Get vertices for both
    v1 = hexagon_vertices_array([pos1], [np.degrees(ang1)])[0]
    v2 = hexagon_vertices_array([pos2], [np.degrees(ang2)])[0]
    
    # Edge normals for hexagon 1
    # Vertices are at angles ang1 + k*pi/3. Edges are between k and k+1.
    # Normal to edge between v_k and v_{k+1} is perpendicular to (v_{k+1}-v_k).
    # For a regular hexagon, the outward normal to the edge connecting vertex k and k+1
    # is at angle ang1 + (k+0.5)*pi/3.
    
    normals1 = ang1 + (np.arange(6) + 0.5) * np.pi / 3
    normals2 = ang2 + (np.arange(6) + 0.5) * np.pi / 3
    
    all_normals = np.concatenate([normals1, normals2])
    
    for n in all_normals:
        nx = np.cos(n)
        ny = np.sin(n)
        
        # Project vertices of hex 1
        proj1 = v1 @ np.array([nx, ny])
        min1, max1 = np.min(proj1), np.max(proj1)
        
        # Project vertices of hex 2
        proj2 = v2 @ np.array([nx, ny])
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for separation
        if max1 < min2 - eps or max2 < min1 - eps:
            return False
            
    return True
