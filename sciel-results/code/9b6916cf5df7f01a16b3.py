# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_normals(angle_deg):
        # Normals for edges of a regular hexagon
        # Vertices at 0, 60, ...
        # Edges connect v_i to v_{i+1}
        # Normal is perpendicular to edge, pointing outwards
        # For flat-top (angle 0), vertices are at 0, 60...
        # Edge 0-1: vector (cos60-cos0, sin60-sin0) = (0.5-1, 0.866-0) = (-0.5, 0.866)
        # Normal: (0.866, 0.5) -> angle 30 deg.
        # So normals are at 30, 90, 150, 210, 270, 330.
        base_normals = np.radians(np.arange(30, 360, 60))
        rot = np.radians(angle_deg)
        angles = base_normals + rot
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def check_disjoint(v1, v2, n1, n2):
        # v1, v2: (6,2) vertices
        # n1, n2: (6,2) normals
        # SAT check
        # Project v1 onto n2 and v2 onto n1? No, project both onto all normals.
        # Actually, for regular hexagons, the normals of one are sufficient if we check projections of both?
        # Yes, the set of separating axes for two convex polygons is the union of their edge normals.
        
        # Project v1 onto n1
        p11 = np.dot(v1, n1.T)
        # Project v2 onto n1
        p21 = np.dot(v2, n1.T)
        
        # Project v1 onto n2
        p12 = np.dot(v1, n2.T)
        # Project v2 onto n2
        p22 = np.dot(v2, n2.T)
        
        # Check for separation on n1 axes
        # min(p11) < max(p21) and max(p11) > min(p21) means overlap
        # If max(p11) < min(p21) or max(p21) < min(p11), they are disjoint
        
        eps = 1e-9
        for i in range(6):
            if np.max(p11[:, i]) < np.min(p21[:, i]) - eps or np.max(p21[:, i]) < np.min(p11[:, i]) - eps:
                return True
        for i in range(6):
            if np.max(p12[:, i]) < np.min(p22[:, i]) - eps or np.max(p22[:, i]) < np.min(p12[:, i]) - eps:
                return True
        return False

    def all_disjoint(inner_positions):
        # inner_positions: list of (x, y, angle)
        # Precompute vertices and normals
        verts = []
        norms = []
        for x, y, a in inner_positions:
            v = hexagon_vertices(x, y, 1.0, a)
            n = get_normals(a)
            verts.append(v)
            norms.append(n)
            
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint(verts[i], verts[j], norms[i], norms[j]):
                    return False
        return True

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        # Vectorized point in hexagon check
        # Transform point to local frame
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        dists = local_x * nx + local_y * ny
        return np.all(dists <= apothem + 1e-9)

    def all_inside_outer(inner_positions, outer_center, outer_side, outer_angle):
        # Check if all vertices of all inner hexagons are inside the outer hexagon
        # Or more efficiently, check if the "support functions" align?
        # Simpler: Check vertices.
        for x, y, a in inner_positions:
            v = hexagon_vertices(x, y, 1.0, a)
            for vx, vy in v:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        return True

    def objective(params):
        # Parse params
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        ox = params[3*N]
        oy = params[3*N+1]
        os = max(params[3*N+2], 0.1)
        oa = params[3*N+3]
        
        # Calculate penalty
        penalty = 0.0
        # 1. Check disjointness
        if not all_disjoint(inner):
            # Count overlaps for smoother gradient?
            # For now, just a large penalty
            penalty += 10.0
        else:
            # 2. Check containment
            if not all_inside_outer(inner, (ox, oy), os, oa):
                # Calculate how much it sticks out?
                # Simple penalty: 10.0
                penalty += 10.0
                
        return os + penalty

    def is_valid(inner, oc, os, oa):
        return all_disjoint(inner) and all_inside_outer(inner, oc, os, oa)

    # Initial Configuration: Hexagonal packing
    # 12 hexagons.
    # A 3x4 grid of hexagons?
    # Or a hexagonal cluster?
    # Let's try a grid that is known to be compact.
    # Side length 1.
    # Distance between centers in same row: 2 * side * cos(30) = sqrt(3) ~ 1.732?
    # No, for flat-top hexagons:
    # Width = 2. Height = sqrt(3).
    # If we place them in a grid:
    # Row 0: y=0, x = 0, 2, 4, 6? No, they touch at edges.
    # Distance between centers for touching flat-top hexagons:
    # Horizontal: 2 * side * cos(30) * 2? No.
    # Let's just use a safe grid and let DE tighten it.
    
    # Better initial guess:
    # 1 center, 6 around it, 5 more?
    # Or just a 3x4 grid with some offset.
    
    # Let's use a standard hex lattice subset.
    # dx = 1.5, dy = sqrt(3)/2 * 1.5?
    # For unit hexagons:
    # Center to center distance for touching is sqrt(3) (if edge to edge) or 2 (if vertex to vertex)?
    # Apothem = sqrt(3)/2.
    # If flat-top, horizontal distance between centers of touching hexagons is 2 * apothem * 2? No.
    # The distance between parallel edges is sqrt(3).
    # So if we align them, x spacing is 2 * apothem * 2? No.
    # Let's just use a grid with spacing 2.1 in x and 1.8 in y to be safe, then optimize.
    
    x_coords = [0, 2.1, 4.2, 6.3]
    y_coords = [0, 1.8, 3.6]
    
    initial_positions = []
    for y in y_coords:
        for x in x_coords:
            initial_positions.append([x, y, 0.0])
            
    # Center the configuration
    xs = [p[0] for p in initial_positions]
    ys = [p[1] for p in initial_positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    for p in initial_positions:
        p[0] -= cx
        p[1] -= cy
        
    outer_side_init = 5.0
    outer_center_init = [0.0, 0.0]
    outer_angle_init = 0.0
    
    params = []
    for p in initial_positions:
        params.extend([p[0], p[1], p[2]])
    params.extend([outer_center_init[0], outer_center_init[1], outer_side_init, outer_angle_init])
    
    # Check initial validity
    inner_init = [(p[0], p[1], p[2]) for p in initial_positions]
    if not is_valid(inner_init, outer_center_init, outer_side_init, outer_angle_init):
         # Expand outer side until valid
         while not all_inside_outer(inner_init, outer_center_init, outer_side_init, outer_angle_init):
             outer_side_init += 0.1
             if outer_side_init > 10: break
         params[3*N+2] = outer_side_init

    best_params = params.copy()
    best_score = objective(params)
    
    # Optimization using Differential Evolution
    # Bounds for inner positions
    # Assume outer side is around 4-6.
    # So x,y in [-3, 3]
    bounds = []
    for i in range(N):
        bounds.append((-4, 4)) # x
        bounds.append((-4, 4)) # y
        bounds.append((0, 60)) # angle (symmetry)
    bounds.append((-1, 1)) # ox
    bounds.append((-1, 1)) # oy
    bounds.append((3.0, 6.0)) # os (min side to fit 12 hexagons is > 3)
    bounds.append((0, 60)) # oa
    
    # Run DE
    # Time limit is tight.
    # Use a small population?
    # DE is stochastic.
    
    try:
        # Use a wrapper to enforce time limit?
        # DE doesn't have a direct time limit, but we can check time in callback?
        # Or just run a limited number of generations.
        
        # Let's try a quick DE run.
        # maxiter=50, popsize=10 might be fast enough?
        # 12*3 + 4 = 40 variables.
        # popsize 10 -> 400 evaluations.
        # 50 iters -> 20000 evaluations.
        # Each eval is fast now?
        # Let's estimate time per eval.
        # SAT check is vectorized. Should be < 1ms.
        # 20000 * 1ms = 20s. Acceptable.
        
        res = differential_evolution(objective, bounds, maxiter=30, popsize=10, seed=42, tol=0.01, polish=False)
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    except Exception as e:
        pass
    
    # Refine with Nelder-Mead if time permits
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 0.005, 'fatol': 0.005, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass

    final_params = best_params
    inner = []
    for i in range(N):
        inner.append((final_params[3*i], final_params[3*i+1], final_params[3*i+2]))
    oc = (final_params[3*N], final_params[3*N+1])
    os = final_params[3*N+2]
    oa = final_params[3*N+3]
    
    # Final validation
    if not is_valid(inner, oc, os, oa):
        # Fallback to initial valid guess
        inner = [(p[0], p[1], p[2]) for p in initial_positions]
        oc = tuple(outer_center_init)
        os = outer_side_init
        oa = outer_angle_init
        if not is_valid(inner, oc, os, oa):
             while not all_inside_outer(inner, oc, os, oa):
                 os += 0.1
                 if os > 10: break

    return inner, oc, os, oa
# EVOLVE-BLOCK-END
