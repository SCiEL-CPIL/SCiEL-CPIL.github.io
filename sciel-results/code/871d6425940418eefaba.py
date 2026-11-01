# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
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

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth surrogate for compute_required_side using log-sum-exp soft-max.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    proj_arr = np.array(projections)
    max_proj = float(np.log(np.sum(np.exp(beta * proj_arr))) / beta)
    
    return 2.0 * max_proj / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Define several high-symmetry topologies (axial coordinates) for 11 hexagons.
    # These are subsets of the hexagonal lattice.
    # We use a small scale factor (1 + eps) to ensure strict disjointness if they touch.
    # Actually, for SAT, touching is usually considered overlap if eps is positive.
    # We will use a scale of 1.001 to be safe, or handle eps in check.
    # Let's stick to standard lattice coords and rely on check_all_disjoint with small eps.
    # If they are exactly touching, dist = sqrt(3). min_dist for penalty is sqrt(3).
    # For SAT, if we use eps=1e-6, touching (dist=sqrt(3)) might be flagged as overlap if numerical errors make it slightly less.
    # To be robust, we scale the cluster by 1.0001.
    
    topologies = [
        # 1. Center + Ring 1 (6) + 4 from Ring 2 (symmetric spread)
        # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Ring 2 candidates: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
        # Selection: (2,0), (1,1), (0,2), (-1,2) -> Top right arc
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2)
        ],
        # 2. Center + Ring 1 (6) + 4 from Ring 2 (different arc)
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (2, -1), (1, -2), (0, -2)
        ],
        # 3. Center + Ring 1 (6) + 3 from Ring 2 (1,1), (0,2), (-1,2) + 1 more? No, need 11.
        # Let's try a "compact" shape.
        # (0,0) + 6 ring + (2,0), (1,1), (0,2) is 10. Add (-1,2)? That's the one above.
        # How about (0,0) + 6 ring + (2,0), (1,1), (1,-1) is 10. Add (2,-1)?
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (2, -1), (1, -2)
        ],
        # 4. A slightly different shape: (0,0) + 6 ring + (2,0), (0,2), (-2,2), (-2,0)
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (0, 2), (-2, 2), (-2, 0)
        ],
        # 5. "Star" shape?
        # (0,0) + 6 ring + (2,0), (0,2), (-2,2), (-2,0), (0,-2) -> 12? No, 1+6+5=12.
        # Remove one? (2,0), (0,2), (-2,2), (-2,0) is 4. Total 11. This is Topology 4.
        
        # 6. Another variation: (0,0) + 6 ring + (2,0), (1,1), (0,2), (1,-1)
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (1, -1) # Wait, (1,-1) is in ring 1. Duplicate!
        ],
        # Correcting Topology 6:
        # (0,0) + 6 ring + (2,0), (1,1), (0,2), (-1,2) is Topology 1.
        # Let's try: (0,0) + 6 ring + (2,0), (1,1), (0,2), (2,-1)
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (2, -1)
        ]
    ]
    
    # Clean topologies: remove duplicates and ensure size 11
    clean_topologies = []
    seen = set()
    for topo in topologies:
        if len(topo) != 11:
            continue
        # Sort and freeze for uniqueness
        frozen = tuple(sorted(topo))
        if frozen not in seen:
            seen.add(frozen)
            clean_topologies.append(topo)
            
    # Add a few more random-ish but symmetric ones if we have time?
    # Let's stick to these for now.
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0, 0)
    best_outer_angle = 0.0
    
    scale = 1.0001 # Slight expansion to ensure disjointness for SAT
    
    for idx, axial_coords in enumerate(clean_topologies):
        if time.time() > deadline - 30:
            break
            
        # Generate initial centers
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
        
        # We will optimize:
        # 1. Global rotation of the cluster (theta)
        # 2. Container center (cx, cy)
        # 3. Container angle (alpha)
        
        # To reduce search space, we can fix cx, cy to the centroid of the cluster for each theta.
        # Then optimize theta and alpha.
        
        def objective_cluster_container(theta, alpha):
            # Rotate cluster
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            rotated_centers = []
            for x, y in centers:
                rx = x * cos_t - y * sin_t
                ry = x * sin_t + y * cos_t
                rotated_centers.append((rx, ry))
            
            # Centroid
            cx = sum(p[0] for p in rotated_centers) / len(rotated_centers)
            cy = sum(p[1] for p in rotated_centers) / len(rotated_centers)
            
            # Inner hexagons have angle = theta (since they are rotated with the cluster)
            # Note: The orientation of the hexagon itself matters.
            # If we rotate the position, we should rotate the hexagon too to maintain the relative geometry.
            inner_list = [[rx, ry, math.degrees(theta)] for rx, ry in rotated_centers]
            
            side = compute_required_side(inner_list, (cx, cy), alpha)
            return side
        
        # Optimize theta and alpha
        # Initial guesses
        best_theta = 0.0
        best_alpha = 0.0
        best_val = float('inf')
        
        # Grid search for theta and alpha to find good initial points
        # Theta range: 0 to 60 (symmetry of hexagon)
        # Alpha range: 0 to 60
        
        # We can use Nelder-Mead on (theta, alpha)
        # Or a simple grid search followed by NM.
        
        grid_theta = [0, 10, 20, 30, 40, 50]
        grid_alpha = [0, 10, 20, 30, 40, 50]
        
        for gt in grid_theta:
            for ga in grid_alpha:
                val = objective_cluster_container(math.radians(gt), math.radians(ga))
                if val < best_val:
                    best_val = val
                    best_theta = math.radians(gt)
                    best_alpha = math.radians(ga)
        
        # Refine with Nelder-Mead
        def obj_nm(params):
            return objective_cluster_container(params[0], params[1])
        
        init_params = [best_theta, best_alpha]
        
        try:
            res = minimize(obj_nm, init_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x[0]
                best_alpha = res.x[1]
        except Exception:
            pass
        
        # Final check and update
        if best_val < best_side:
            # Construct the final inner list
            cos_t = math.cos(best_theta)
            sin_t = math.sin(best_theta)
            final_centers = []
            for x, y in centers:
                rx = x * cos_t - y * sin_t
                ry = x * sin_t + y * cos_t
                final_centers.append((rx, ry))
            
            cx = sum(p[0] for p in final_centers) / len(final_centers)
            cy = sum(p[1] for p in final_centers) / len(final_centers)
            
            final_inner = [[rx, ry, math.degrees(best_theta)] for rx, ry in final_centers]
            
            # Verify disjointness
            if check_all_disjoint(final_inner, eps=1e-5):
                best_side = best_val
                best_inner = final_inner
                best_outer_center = (cx, cy)
                best_outer_angle = math.degrees(best_alpha)
            else:
                # If disjointness fails, try with a slightly larger scale or different eps?
                # For now, skip this topology or adjust scale.
                pass

    # Fallback if no topology worked (should not happen)
    if best_inner is None:
        # Use the first topology with a safe scale
        axial_coords = clean_topologies[0]
        scale = 1.01
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
        best_inner = [[x, y, 0.0] for x, y in centers]
        cx = sum(p[0] for p in centers) / len(centers)
        cy = sum(p[1] for p in centers) / len(centers)
        best_outer_center = (cx, cy)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    # Final polish on the container for the best topology found
    # Optimize cx, cy, alpha for the fixed best_inner
    def objective_polish(params):
        cx, cy, alpha = params
        return compute_required_side(best_inner, (cx, cy), alpha)
    
    init_polish = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    try:
        res = minimize(objective_polish, init_polish, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < best_side:
            best_side = res.fun
            best_outer_center = (res.x[0], res.x[1])
            best_outer_angle = res.x[2]
    except Exception:
        pass

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
