<<<<<<< SEARCH
def sat_min_separation(c1, c2):
    """
    Compute minimum separation between two hexagons using SAT.
    Returns positive if separated, negative if overlapping.
    Also returns the axis of minimum separation.
    """
    v1 = hexagon_vertices((c1[0], c1[1]), 1.0, c1[2])
    v2 = hexagon_vertices((c2[0], c2[1]), 1.0, c2[2])
    
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            n = np.array([-edge[1], edge[0]]) / edge_len
            axes.append(n)
    
    min_sep = float('inf')
    best_axis = None
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep < min_sep:
            min_sep = sep
            best_axis = axis
    
    return min_sep, best_axis

def _compute_hex_axes(cx, cy, angle_deg):
    """Compute the 6 edge normals of a unit hexagon at given center/angle."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # Precompute cos/sin for all 6 edges
    thetas = a + np.radians(60 * np.arange(6))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    # Edge i goes from vertex i to vertex i+1
    # Normal is perpendicular to edge, pointing outward
    # Edge direction: (cos(theta_{i+1}) - cos(theta_i), sin(theta_{i+1}) - sin(theta_i))
    # For unit hexagon, edge length = 1, so normal = (-dy, dx) normalized
    dx = cos_t[1:] - cos_t[:-1]
    dy = sin_t[1:] - sin_t[:-1]
    # Wrap around for last edge
    dx = np.append(dx, cos_t[0] - cos_t[-1])
    dy = np.append(dy, sin_t[0] - sin_t[-1])
    # Normal = (-dy, dx), already unit length for unit hexagon edges
    normals = np.column_stack([-dy, dx])
    return normals

def _compute_hex_vertex_projections(cx, cy, angle_deg, axes):
    """Compute projections of hexagon vertices onto given axes. Returns (min_proj, max_proj) per axis."""
    a = np.radians(angle_deg)
    thetas = a + np.radians(60 * np.arange(6))
    vx = cx + np.cos(thetas)
    vy = cy + np.sin(thetas)
    # Project vertices onto each axis
    # axes shape: (12, 2), vertices: (6, 2)
    # proj[i, j] = dot(vertex_j, axis_i)
    proj = axes[:, 0] * vx[None, :] + axes[:, 1] * vy[None, :]
    min_proj = proj.min(axis=1)
    max_proj = proj.max(axis=1)
    return min_proj, max_proj

def smooth_min_separation(c1, c2, beta=50.0):
    """
    Compute a smooth approximation of minimum separation using soft-min.
    Returns (smooth_sep, smooth_axis_approx).
    """
    axes1 = _compute_hex_axes(c1[0], c1[1], c1[2])
    axes2 = _compute_hex_axes(c2[0], c2[1], c2[2])
    all_axes = np.vstack([axes1, axes2])
    
    min1, max1 = _compute_hex_vertex_projections(c1[0], c1[1], c1[2], all_axes)
    min2, max2 = _compute_hex_vertex_projections(c2[0], c2[1], c2[2], all_axes)
    
    sep1 = min1 - max2
    sep2 = min2 - max1
    seps = np.maximum(sep1, sep2)
    
    min_val = np.min(seps)
    exps = np.exp(beta * (seps - min_val))
    soft_min = min_val - np.log(np.sum(exps)) / beta
    
    weights = exps / np.sum(exps)
    approx_axis = np.sum(weights[:, np.newaxis] * all_axes, axis=0)
    approx_axis_norm = np.linalg.norm(approx_axis)
    if approx_axis_norm > 1e-12:
        approx_axis = approx_axis / approx_axis_norm
    
    return soft_min, approx_axis

def smooth_required_outer_side(centers, outer_center, outer_angle, beta=30.0):
    """
    Compute a smooth approximation of the required outer side length.
    """
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    # Outer hexagon normals (in local frame)
    outer_thetas = np.radians(60 * np.arange(6))
    outer_normals = np.column_stack([np.cos(outer_thetas), np.sin(outer_thetas)])
    
    d_values = []
    
    for (px, py, pa) in centers:
        a = np.radians(pa)
        thetas = a + np.radians(60 * np.arange(6))
        vx = px + np.cos(thetas)
        vy = py + np.sin(thetas)
        
        # Relative to outer center
        rx = vx - outer_center[0]
        ry = vy - outer_center[1]
        
        # Rotate to outer frame
        rx_local = rx * cos_a - ry * sin_a
        ry_local = rx * sin_a + ry * cos_a
        
        # Project onto outer normals
        d = np.abs(rx_local[:, None] * outer_normals[None, :, 0] + ry_local[:, None] * outer_normals[None, :, 1])
        d_values.append(d.flatten())
    
    d_arr = np.concatenate(d_values)
    min_d = np.min(d_arr)
    exps = np.exp(beta * (d_arr - min_d))
    soft_max = min_d + np.log(np.sum(exps)) / beta
    
    return soft_max / (np.sqrt(3) / 2)

def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    max_extent = 0.0
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    for (px, py, pa) in centers:
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        for v in inner_verts:
            rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rx * nx + ry * ny)
                max_extent = max(max_extent, d)
    
    return max_extent / (np.sqrt(3) / 2)
=======
def sat_min_separation(c1, c2):
    """
    Compute minimum separation between two hexagons using SAT.
    Returns positive if separated, negative if overlapping.
    Also returns the axis of minimum separation.
    """
    v1 = hexagon_vertices((c1[0], c1[1]), 1.0, c1[2])
    v2 = hexagon_vertices((c2[0], c2[1]), 1.0, c2[2])
    
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            n = np.array([-edge[1], edge[0]]) / edge_len
            axes.append(n)
    
    min_sep = float('inf')
    best_axis = None
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep < min_sep:
            min_sep = sep
            best_axis = axis
    
    return min_sep, best_axis

def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    max_extent = 0.0
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    for (px, py, pa) in centers:
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        for v in inner_verts:
            rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rx * nx + ry * ny)
                max_extent = max(max_extent, d)
    
    return max_extent / (np.sqrt(3) / 2)
>>>>>>> REPLACE

