<<<<<<< SEARCH
    # Generate inner hexagon positions
    # Each inner hexagon has side length 1, angle 0
    # We'll place them in a hexagonal lattice
    
    inner_positions = hex_lattice_positions(N)
    
    # For each inner hexagon, get its vertices
    # A regular hexagon with side s=1, center (cx, cy), angle theta has vertices:
    # (cx + s*cos(theta + k*pi/3), cy + s*sin(theta + k*pi/3)) for k=0..5
    
    def get_hexagon_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon."""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for k in range(6):
            theta = angle_rad + k * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for (cx, cy) in inner_positions:
        verts = get_hexagon_vertices(cx, cy, 0.0)
        all_vertices.extend(verts.tolist())
    
    all_vertices = np.array(all_vertices)
    
    # For a given outer hexagon rotation, compute the minimum side length
    # that contains all vertices
    
    def compute_min_side_length(outer_rotation_deg):
        """Compute minimum side length of outer hexagon with given rotation."""
        rot_rad = np.radians(outer_rotation_deg)
        
        # Outer hexagon has 6 normal directions at angles:
        # rot + k*pi/3 for k=0..5
        # The distance from center to each side is side_length * sqrt(3)/2
        # For a vertex at distance d from center in direction phi,
        # the required side length is:
        # side = max over all vertices and all normals of:
        #   (vertex dot normal) / (sqrt(3)/2)
        # where normal is the unit normal pointing outward
        
        # Actually, for a regular hexagon with side s, the apothem (inradius) is s*sqrt(3)/2
        # A point (x,y) is inside if for all 6 normals n_i:
        #   (x,y) dot n_i <= apothem
        # So apothem >= max_i (x,y) dot n_i for all points
        # And side = apothem / (sqrt(3)/2)
        
        max_projection = 0
        for k in range(6):
            normal_angle = rot_rad + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            projections = all_vertices @ normal
            max_proj = np.max(np.abs(projections))
            max_projection = max(max_projection, max_proj)
        
        apothem = max_projection
        side_length = apothem / (sqrt3 / 2)
        return side_length
=======
    # Generate inner hexagon positions
    # Each inner hexagon has side length 1, angle 0
    # We'll place them in a hexagonal lattice
    
    inner_positions = hex_lattice_positions(N)
    
    # For each inner hexagon, get its vertices
    # A regular hexagon with side s=1, center (cx, cy), angle theta has vertices:
    # (cx + s*cos(theta + k*pi/3), cy + s*sin(theta + k*pi/3)) for k=0..5
    
    def get_hexagon_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon."""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for k in range(6):
            theta = angle_rad + k * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for (cx, cy) in inner_positions:
        verts = get_hexagon_vertices(cx, cy, 0.0)
        all_vertices.extend(verts.tolist())
    
    all_vertices = np.array(all_vertices)
    
    # Center the outer hexagon at the centroid of inner hexagons
    if len(inner_positions) > 0:
        centers = np.array(inner_positions)
        center = (float(np.mean(centers[:, 0])), float(np.mean(centers[:, 1])))
    else:
        center = (0.0, 0.0)
        
    # Subtract center from vertices to work in local coordinates relative to outer hexagon center
    center_vec = np.array(center)
    local_vertices = all_vertices - center_vec

    # For a given outer hexagon rotation, compute the minimum side length
    # that contains all vertices
    
    def compute_min_side_length(outer_rotation_deg):
        """Compute minimum side length of outer hexagon with given rotation."""
        rot_rad = np.radians(outer_rotation_deg)
        
        # Outer hexagon has 6 normal directions at angles:
        # rot + k*pi/3 for k=0..5
        # The distance from center to each side is side_length * sqrt(3)/2
        # A point (x,y) is inside if for all 6 normals n_i:
        #   (x,y) dot n_i <= apothem
        # So apothem >= max_i (x,y) dot n_i for all points
        # And side = apothem / (sqrt(3)/2)
        
        max_projection = 0
        for k in range(6):
            normal_angle = rot_rad + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            projections = local_vertices @ normal
            max_proj = np.max(np.abs(projections))
            max_projection = max(max_projection, max_proj)
        
        apothem = max_projection
        side_length = apothem / (sqrt3 / 2)
        return side_length
>>>>>>> REPLACE

