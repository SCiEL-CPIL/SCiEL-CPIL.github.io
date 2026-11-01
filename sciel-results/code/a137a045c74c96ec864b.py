<<<<<<< SEARCH
def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3
=======
def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons (vectorized).
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    n = len(inner)
    
    # Precompute normals
    normal_angles = np.array([math.radians(30 + i * 60) for i in range(6)])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    # For each inner hexagon, compute 6 vertices
    # Vertex k of hexagon with center (cx,cy), side 1, angle a:
    #   (cx + cos(a + k*pi/3), cy + sin(a + k*pi/3))
    # All 6 vertices for all n hexagons: shape (n, 6, 2)
    
    # angles for vertices: angle_rad + k*pi/3 for k=0..5
    vertex_angle_offsets = np.array([k * math.pi / 3.0 for k in range(6)])  # (6,)
    
    # inner angles in radians: (n,)
    inner_angles = np.array([math.radians(p[2]) for p in inner])  # (n,)
    
    # Full vertex angles: (n, 6)
    full_angles = inner_angles[:, None] + vertex_angle_offsets[None, :]
    
    # Vertex offsets from center: (n, 6, 2)
    cos_va = np.cos(full_angles)  # (n, 6)
    sin_va = np.sin(full_angles)  # (n, 6)
    
    # Centers: (n, 2)
    centers = np.array([[p[0], p[1]] for p in inner])  # (n, 2)
    
    # Vertex positions: (n, 6, 2)
    vx = centers[:, 0:1] + cos_va  # (n, 6)
    vy = centers[:, 1:2] + sin_va  # (n, 6)
    
    # Translate to outer center
    dx = vx - ox  # (n, 6)
    dy = vy - oy  # (n, 6)
    
    # Rotate by outer angle
    lx = dx * cos_a + dy * sin_a  # (n, 6)
    ly = -dx * sin_a + dy * cos_a  # (n, 6)
    
    # Project onto 6 normals: (n, 6, 6)
    # proj[i, k, j] = lx[i,k]*nx[j] + ly[i,k]*ny[j]
    proj = lx[:, :, None] * normals[None, None, 0] + ly[:, :, None] * normals[None, None, 1]
    
    max_projection = np.max(proj)
    
    return 2.0 * max_projection / SQRT3
>>>>>>> REPLACE

