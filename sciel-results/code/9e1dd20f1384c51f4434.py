import numpy as np
import time
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _sdf_hex_point(p, center, side, angle_deg):
    """
    Signed distance from point p to a regular hexagon.
    Positive if outside, negative if inside.
    """
    x, y = p
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    
    # Transform to hexagon local frame
    dx = x - cx
    dy = y - cy
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    
    # Hexagon with side 's' has apothem a = s * sqrt(3)/2
    # Vertices are at distance s from center.
    # In local frame, the hexagon is defined by 6 half-planes.
    # Due to symmetry, we can map to one sector.
    
    # Actually, a robust SDF for a regular polygon can be computed by finding the closest point on the boundary.
    # For a hexagon, we can check the 6 edges.
    # But a faster analytic approach:
    # The hexagon is the intersection of 6 strips.
    # Distance to the hexagon is the max of the signed distances to the 6 lines, 
    # BUT only if we are outside the "core" region? No, that's for union.
    # For intersection (convex poly), the signed distance is the minimum of the signed distances to the supporting hyperplanes?
    # No, that's not right either. The distance to a convex set is the minimum distance to any point in the set.
    
    # Let's stick to a robust vertex/edge check for SDF, or use the property that for a regular hexagon,
    # we can rotate the point to the first sector (0 to 30 deg) and compute distance to the two bounding edges.
    
    # Normalize angle to [0, 360)
    # Find which sector the point is in.
    # Sectors are centered at 0, 60, 120, 180, 240, 300.
    # Edges are at 30, 90, 150, 210, 270, 330.
    
    # Let's compute distance to each of the 6 edges and take the minimum.
    # This is O(6) per point.
    
    verts = _hex_vertices((0,0), side, 0) # Local frame, center 0, angle 0
    # Wait, the local frame transformation already rotated the point. So we use a hexagon at 0,0 with angle 0.
    # Vertices of unit hexagon (side=1) at angle 0:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Scaled by 'side'.
    
    # Edges:
    # e0: (1,0) to (0.5, h)
    # e1: (0.5, h) to (-0.5, h)
    # ...
    
    h = side * np.sqrt(3) / 2.0
    v0 = np.array([side, 0])
    v1 = np.array([side/2, h])
    v2 = np.array([-side/2, h])
    v3 = np.array([-side, 0])
    v4 = np.array([-side/2, -h])
    v5 = np.array([side/2, -h])
    
    verts = [v0, v1, v2, v3, v4, v5]
    
    min_dist_sq = np.inf
    inside = True
    
    # Check if inside using cross products (CCW order)
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i+1)%6]
        # Vector p1->p2
        e = p2 - p1
        # Vector p1->p
        v = np.array([lx, ly]) - p1
        # Cross product z-component
        cross = e[0]*v[1] - e[1]*v[0]
        if cross < 0:
            inside = False
            break
    
    if inside:
        # If inside, distance is negative of the minimum distance to boundary.
        # We still need to compute min dist to boundary.
        pass
    
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i+1)%6]
        # Project point onto segment p1-p2
        ab = p2 - p1
        ap = np.array([lx, ly]) - p1
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0.0, min(1.0, t))
        proj = p1 + t * ab
        dist = np.linalg.norm(np.array([lx, ly]) - proj)
        min_dist_sq = min(min_dist_sq, dist)
        
    dist = np.sqrt(min_dist_sq)
    if inside:
        return -dist
    else:
        return dist

def _sdf_hex_hex(verts1, center2, side2, angle2):
    """
    Approximate SDF between two hexagons.
    Uses vertices of hex1 and checks distance to hex2.
    Also checks vertices of hex2 to hex1? 
    For a robust SDF, we need min over all pairs.
    However, for optimization, checking vertices of one against the other is often sufficient 
    if we also check if centroids are inside? 
    Actually, the minimum distance between two convex polygons occurs at either:
    1. Vertex-Edge
    2. Edge-Edge (parallel)
    
    Checking all vertices of hex1 against hex2 and all vertices of hex2 against hex1 covers most cases.
    Edge-Edge parallel cases are measure zero in continuous optimization, so vertex checks should suffice for gradient descent.
    """
    min_d = np.inf
    # Check vertices of hex1 against hex2
    for v in verts1:
        d = _sdf_hex_point(v, center2, side2, angle2)
        min_d = min(min_d, d)
        
    # Check vertices of hex2 against hex1
    verts2 = _hex_vertices(center2, side2, angle2)
    # We need SDF of hex1. We can compute it by checking distance from v2 to hex1.
    # But _sdf_hex_point expects a regular hexagon defined by center, side, angle.
    # So we can just call it.
    for v in verts2:
        # We need the center, side, angle of hex1. 
        # This function doesn't have them. 
        # We should pass the full definition of hex1 as well.
        pass
        
    # Let's refactor: _sdf_hex_hex(verts1, params1, params2)
    # But for speed in the loop, let's just use the vertex check for hex1 vs hex2 definition.
    # And assume symmetry? No.
    
    # Let's create a helper that takes two full definitions.
    return min_d

def _sdf_hex_hex_full(verts1, c1, s1, a1, verts2, c2, s2, a2):
    min_d = np.inf
    for v in verts1:
        d = _sdf_hex_point(v, c2, s2, a2)
        min_d = min(min_d, d)
    for v in verts2:
        d = _sdf_hex_point(v, c1, s1, a1)
        min_d = min(min_d, d)
    return min_d

def _smooth_collision_penalty(params, outer_side, outer_center, outer_angle):
    """
    Compute a smooth penalty for collisions.
    params: (N, 3) array of [x, y, angle]
    """
    penalty = 0.0
    centers = params[:, :2]
    angles = params[:, 2]
    sides = np.ones(N)
    
    verts_list = [_hex_vertices(centers[i], 1.0, angles[i]) for i in range(N)]
    
    # Check inner-inner
    for i in range(N):
        for j in range(i+1, N):
            d = _sdf_hex_hex_full(verts_list[i], centers[i], 1.0, angles[i], 
                                  verts_list[j], centers[j], 1.0, angles[j])
            if d < 0:
                # Overlap. Penalty is quadratic in penetration depth.
                penalty += (d * d) * 100.0
                
    # Check inner-outer
    # Outer hexagon is defined by outer_center, outer_side, outer_angle.
    # SDF from inner vertex to outer hexagon should be >= 0.
    # _sdf_hex_point returns negative if inside.
    # So we want d_outer >= 0.
    # Penalty for d_outer < 0.
    
    for i in range(N):
        for v in verts_list[i]:
            d = _sdf_hex_point(v, outer_center, outer_side, outer_angle)
            if d < 0:
                penalty += (d * d) * 100.0
                
    return penalty

def _objective(x, outer_center, outer_angle):
    """
    Objective: Minimize outer_side + penalty.
    x: 3N vector.
    We fix outer_center and outer_angle for this step? 
    Or optimize them too? 
    The task allows optimizing outer side, center, angle.
    Let's optimize x (inner params) and outer_side. 
    Fix outer_center to (0,0) and outer_angle to 0 for simplicity in this first pass, 
    or include them in x?
    Including them adds 3 variables.
    Let's try optimizing inner params and outer_side, keeping outer_center=(0,0) and outer_angle=0.
    Actually, rotating the outer hexagon is equivalent to rotating the inner ones relative to it, 
    but since inner ones can rotate freely, fixing outer to 0 is WLOG for the relative geometry, 
    EXCEPT that the outer hexagon shape is fixed (regular). 
    Wait, if I rotate the outer hexagon, the constraints rotate. 
    Since the inner hexagons can rotate arbitrarily, the global rotation of the whole system doesn't matter.
    So we can fix outer_angle = 0 and outer_center = (0,0).
    We only need to optimize inner positions/angles and the outer side length.
    
    However, the "side length" is not a parameter in x. 
    We can define the objective as:
    f(x) = S(x) + Penalty(x, S(x))
    where S(x) is the minimum side length required to contain the hexagons (without collision penalty).
    But S(x) is non-smooth (max of projections).
    
    Alternative:
    Add outer_side as a variable in x.
    x = [x1, y1, a1, ..., xN, yN, aN, S]
    Objective = S + Penalty(x, S)
    Penalty includes containment in hexagon of side S.
    """
    params = x[:-1].reshape(N, 3)
    S = x[-1]
    
    # S must be positive
    if S < 1.0:
        return 1000.0
    
    penalty = _smooth_collision_penalty(params, S, (0.0, 0.0), 0.0)
    
    return S + penalty

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_params = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Initial guess: Hexagonal ring
    # Center at 0,0. 6 around it. 5 more?
    # Let's use a lattice.
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # Lattice vectors: (sqrt(3), 0) and (sqrt(3)/2, 1.5)
    
    pts = []
    # Generate a grid of lattice points
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * np.sqrt(3) + j * np.sqrt(3) / 2
            y = j * 1.5
            pts.append([x, y, 0.0])
            
    # Select 12 points closest to origin
    pts_arr = np.array(pts)
    dists = np.sum(pts_arr**2, axis=1)
    sorted_idx = np.argsort(dists)
    
    # Greedy selection to avoid initial overlaps? 
    # Lattice points are separated by sqrt(3), which is exactly touching.
    # So any 12 distinct lattice points are non-overlapping (touching is allowed? "disjoint" usually means no interior overlap. Touching is OK).
    # Let's pick the first 12.
    initial_pts = pts_arr[sorted_idx[:12]]
    
    # Add S estimate
    # Max radius of these points:
    max_r = np.max(np.linalg.norm(initial_pts[:, :2], axis=1))
    S_init = max_r * 2 / np.sqrt(3) * 1.1 # 10% margin
    
    x0 = np.append(initial_pts.flatten(), S_init)
    
    # Bounds
    # Positions: -10 to 10
    # Angles: -180 to 180
    # S: 1 to 20
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((-180, 180))
    bounds.append((1.0, 20.0))
    
    # Run L-BFGS-B
    try:
        res = minimize(
            _objective,
            x0,
            args=((0.0, 0.0), 0.0),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_side:
            x_opt = res.x
            params_opt = x_opt[:-1].reshape(N, 3)
            S_opt = x_opt[-1]
            
            # Verify and refine
            # We need to check if this is valid.
            # If valid, we can try to reduce S further by projecting?
            # Or just run another optimization with a tighter S bound?
            
            # Let's do a final polish: 
            # Fix positions, minimize S subject to constraints?
            # Or just accept this result.
            
            # Check validity
            inner_hexes = [[params_opt[i][0], params_opt[i][1], params_opt[i][2]] for i in range(N)]
            is_valid, _ = _check_validity(inner_hexes, S_opt, (0.0, 0.0), 0.0)
            
            if is_valid:
                best_side = S_opt
                best_params = inner_hexes
                best_outer_center = (0.0, 0.0)
                best_outer_angle = 0.0
            else:
                # If invalid, try to repair by increasing S slightly or running again
                # For now, fallback to seed-like result if this fails?
                # Let's try one more run with a different initial guess (random)
                pass
    except Exception as e:
        pass
        
    # Second attempt with random initialization if first failed or to find better
    if time.time() - start_time < TIME_LIMIT - 20:
        np.random.seed(123)
        for _ in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            # Random cluster
            x_rand = np.random.uniform(-5, 5, N*3)
            x_rand[2::3] = np.random.uniform(-180, 180, N) # Angles
            S_rand = 6.0
            x_rand_full = np.append(x_rand, S_rand)
            
            try:
                res = minimize(
                    _objective,
                    x_rand_full,
                    args=((0.0, 0.0), 0.0),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                if res.fun < best_side:
                    x_opt = res.x
                    params_opt = x_opt[:-1].reshape(N, 3)
                    S_opt = x_opt[-1]
                    
                    inner_hexes = [[params_opt[i][0], params_opt[i][1], params_opt[i][2]] for i in range(N)]
                    is_valid, _ = _check_validity(inner_hexes, S_opt, (0.0, 0.0), 0.0)
                    
                    if is_valid:
                        best_side = S_opt
                        best_params = inner_hexes
                        best_outer_center = (0.0, 0.0)
                        best_outer_angle = 0.0
                        
            except:
                pass

    if best_params is not None:
        return (best_params, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check containment
    for iv in inner_verts:
        for v in iv:
            inside = True
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k+1)%6]
                cross = (b[0]-a[0])*(v[1]-a[1]) - (b[1]-a[1])*(v[0]-a[0])
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False, "outside"
                
    # Check disjointness
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _sat_overlap(inner_verts[i], inner_verts[j]):
                return False, "overlap"
                
    return True, "valid"

def _sat_overlap(v1, v2):
    """
    Check if two convex polygons overlap using Separating Axis Theorem.
    """
    axes = np.vstack([_get_axes(v1), _get_axes(v2)])
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False
            
    return True

def _get_axes(hex_verts):
    """Get the 6 edge normals (axes) of a hexagon."""
    axes = []
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm
        axes.append(normal)
    return np.array(axes)
