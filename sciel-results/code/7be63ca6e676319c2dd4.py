<<<<<<< SEARCH
def _smooth_objective(params, all_inner_verts, beta=50.0):
    """
    Smooth surrogate for the minimum enclosing hexagon side length.
    params: [dx_0, dy_0, ..., dx_10, dy_10, outer_cx, outer_cy, outer_angle_deg]
    Uses log-sum-exp to smooth the max over all vertex-normal pairs.
    """
    n_hex = N
    # Extract parameters
    inner_deltas = params[:2*n_hex]
    outer_cx = params[2*n_hex]
    outer_cy = params[2*n_hex+1]
    outer_angle_deg = params[2*n_hex+2]
    
    # Compute all vertices of inner hexagons with small displacements
    # Base positions are stored globally
    global _base_positions
    all_verts = []
    for i in range(n_hex):
        bx, by = _base_positions[i]
        dx, dy = inner_deltas[2*i], inner_deltas[2*i+1]
        cx, cy = bx + dx, by + dy
        verts = _get_hex_vertices(cx, cy, 0.0)
        all_verts.extend(verts)
    
    # Convert to numpy array
    all_verts = np.array(all_verts)
    
    # Outer hexagon normals
    outer_angle_rad = math.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        a = outer_angle_rad + math.pi/6 + math.pi * k / 3
        normals.append((math.cos(a), math.sin(a)))
    normals = np.array(normals)  # shape (6, 2)
    
    # For each vertex v and normal n, compute distance: (v - c) . n
    # We need: max over all (v, n) of (v - c) . n <= R
    # Also: max over all (v, n) of (c - v) . n <= R
    # i.e., |max((v-c).n)| <= R for all n
    
    # Compute all projections
    # all_verts: (M, 2), normals: (6, 2)
    # For each normal, compute max projection and min projection
    R_vals = []
    for nx, ny in normals:
        projections = all_verts[:, 0] * nx + all_verts[:, 1] * ny
        center_proj = outer_cx * nx + outer_cy * ny
        # Max distance from center in direction n
        max_dist = np.max(projections) - center_proj
        min_dist = center_proj - np.min(projections)
        R_vals.append(max_dist)
        R_vals.append(min_dist)
    
    R_vals = np.array(R_vals)
    
    # Smooth max using log-sum-exp
    # max(x) ≈ (1/beta) * log(sum(exp(beta * x)))
    # But we want to minimize the max, so we use the soft-max
    if beta == 0:
        return float(np.max(R_vals)) * 2.0 / math.sqrt(3)
    
    # For numerical stability, shift by max
    R_shifted = R_vals - np.max(R_vals)
    log_sum_exp = np.max(R_vals) + np.log(np.sum(np.exp(beta * R_shifted))) / beta
    R_soft = log_sum_exp
    
    side = R_soft * 2.0 / math.sqrt(3)
    
    # Penalty for negative R (shouldn't happen but just in case)
    if R_soft < 0:
        side += 100.0 * (-R_soft)
    
    return side
=======
def _smooth_objective(params, all_inner_verts, beta=50.0):
    """
    Smooth surrogate for the minimum enclosing hexagon side length.
    params: [dx_0, dy_0, ..., dx_10, dy_10, outer_cx, outer_cy, outer_angle_deg]
    Uses log-sum-exp to smooth the max over all vertex-normal pairs.
    """
    n_hex = N
    inner_deltas = params[:2*n_hex]
    outer_cx = params[2*n_hex]
    outer_cy = params[2*n_hex+1]
    outer_angle_deg = params[2*n_hex+2]
    
    global _base_positions
    all_verts = []
    for i in range(n_hex):
        bx, by = _base_positions[i]
        dx, dy = inner_deltas[2*i], inner_deltas[2*i+1]
        cx, cy = bx + dx, by + dy
        verts = _get_hex_vertices(cx, cy, 0.0)
        all_verts.extend(verts)
    
    all_verts = np.array(all_verts)
    
    outer_angle_rad = math.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        a = outer_angle_rad + math.pi/6 + math.pi * k / 3
        normals.append((math.cos(a), math.sin(a)))
    normals = np.array(normals)
    
    R_vals = []
    for nx, ny in normals:
        projections = all_verts[:, 0] * nx + all_verts[:, 1] * ny
        center_proj = outer_cx * nx + outer_cy * ny
        max_dist = np.max(projections) - center_proj
        min_dist = center_proj - np.min(projections)
        R_vals.append(max_dist)
        R_vals.append(min_dist)
    
    R_vals = np.array(R_vals)
    
    if beta == 0:
        return float(np.max(R_vals)) * 2.0 / math.sqrt(3)
    
    R_shifted = R_vals - np.max(R_vals)
    log_sum_exp = np.max(R_vals) + np.log(np.sum(np.exp(beta * R_shifted))) / beta
    R_soft = log_sum_exp
    
    side = R_soft * 2.0 / math.sqrt(3)
    
    if R_soft < 0:
        side += 100.0 * (-R_soft)
    
    return side

def _exact_objective(params):
    """
    Exact objective: compute the true minimum enclosing hexagon side length
    using LP for the given configuration.
    params: [dx_0, dy_0, ..., dx_10, dy_10, outer_cx, outer_cy, outer_angle_deg]
    """
    n_hex = N
    inner_deltas = params[:2*n_hex]
    outer_cx = params[2*n_hex]
    outer_cy = params[2*n_hex+1]
    outer_angle_deg = params[2*n_hex+2]
    
    global _base_positions
    all_verts = []
    for i in range(n_hex):
        bx, by = _base_positions[i]
        dx, dy = inner_deltas[2*i], inner_deltas[2*i+1]
        cx, cy = bx + dx, by + dy
        verts = _get_hex_vertices(cx, cy, 0.0)
        all_verts.extend(verts)
    
    # Use LP to find exact minimum enclosing hexagon
    res = _find_min_enclosing_hex_side_lp(all_verts, outer_angle_deg)
    if res:
        _, _, side = res
        return side
    
    # Fallback: use the provided outer parameters
    outer_angle_rad = math.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        a = outer_angle_rad + math.pi/6 + math.pi * k / 3
        normals.append((math.cos(a), math.sin(a)))
    
    max_R = 0.0
    for nx, ny in normals:
        for vx, vy in all_verts:
            d = abs((vx - outer_cx) * nx + (vy - outer_cy) * ny)
            max_R = max(max_R, d)
    
    return max_R * 2.0 / math.sqrt(3)
>>>>>>> REPLACE

