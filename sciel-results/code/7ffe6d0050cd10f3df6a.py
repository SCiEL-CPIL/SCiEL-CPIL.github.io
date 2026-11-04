import numpy as np
import time
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0
START_TIME = time.time()

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _sdf_hex_point(p, center, angle_deg, side=1.0):
    """
    Compute the SDF of a point p to a regular hexagon centered at 'center'
    with rotation 'angle_deg' and side length 'side'.
    Negative values mean inside.
    """
    # Transform point to local hexagon frame
    theta = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(-theta), -np.sin(-theta)],
        [np.sin(-theta), np.cos(-theta)]
    ])
    p_local = R @ (p - center)
    
    # Hexagon is defined by 3 pairs of parallel sides.
    # In local frame, normals are at 0, 60, 120 degrees.
    # Distance to each pair of sides is |dot(p, n)| - apothem
    # For a regular hexagon of side s, apothem a = s * sqrt(3)/2.
    apothem = side * np.sqrt(3) / 2.0
    
    # Normals in local frame
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2]
    ])
    
    # Distances to each pair of walls
    # d_i = |dot(p, n_i)| - apothem
    # SDF is max of these distances (since it's the intersection of half-planes)
    # Wait, for a convex polygon, SDF(p) = max_i (dot(p, n_i) - h_i) if p is outside?
    # No, SDF(p) = min_i (dot(n_i, p) - h_i) is for union of half spaces?
    # For intersection of half spaces (convex polytope):
    # A point is inside if dot(p, n_i) <= h_i for all i.
    # Distance to boundary is min_i (h_i - dot(p, n_i)) if inside? No.
    # If inside, distance to boundary is min_i (h_i - dot(p, n_i)).
    # If outside, distance is min_i (dot(p, n_i) - h_i).
    # Actually, for convex polygon defined by half planes n_i . x <= h_i:
    # SDF(x) = min_i (n_i . x - h_i) is NOT correct for inside.
    # If x is inside, n_i . x - h_i is negative. The "distance" to the boundary in the direction of n_i is h_i - n_i . x.
    # The Euclidean distance to the boundary is min_i (h_i - n_i . x) for inside points.
    # For outside points, it is min_i (n_i . x - h_i) ? No, that's not right either if the closest feature is a vertex.
    
    # Let's use the standard SDF for a box/hexagon.
    # For a regular hexagon, the closest point on the boundary can be found by projecting onto the 3 axes.
    # Let q = |p_local| in the hexagonal coordinate system?
    
    # Simpler approach:
    # The hexagon is the intersection of 3 infinite strips.
    # Strip i is defined by |dot(p, n_i)| <= apothem.
    # The SDF of a strip is |dot(p, n_i)| - apothem.
    # The SDF of the intersection of strips is max_i (|dot(p, n_i)| - apothem).
    # This is exact for regular hexagons (and boxes).
    
    dists = np.abs(np.dot(p_local, normals.T)) - apothem
    return np.max(dists)

def _sdf_hex_hex(center1, angle1, center2, angle2, side=1.0):
    """
    Approximate SDF between two hexagons.
    We sample points on the boundary of hexagon 1 and compute their SDF to hexagon 2.
    To make it differentiable and fast, we can just check the 6 vertices of hex1 against hex2,
    and 6 vertices of hex2 against hex1, and take the max penetration.
    However, for optimization, a smooth approximation is better.
    We will use the vertices of both hexagons.
    """
    v1 = _hex_vertices(center1, side, angle1)
    v2 = _hex_vertices(center2, side, angle2)
    
    # Min distance between the two sets of vertices? No, that's not the polygon distance.
    # We need the minimum SDF of v1 points w.r.t hex2, and v2 points w.r.t hex1.
    # If all are positive, they are disjoint.
    # If any are negative, they overlap.
    
    # For a smooth penalty, we can sum the squared max violations.
    # Let's compute the "penetration depth" as the max negative SDF.
    
    # SDF of v1 points to hex2
    sdfs_1_to_2 = np.array([_sdf_hex_point(p, center2, angle2, side) for p in v1])
    # SDF of v2 points to hex1
    sdfs_2_to_1 = np.array([_sdf_hex_point(p, center1, angle1, side) for p in v2])
    
    # The minimum distance between the two polygons is approximated by:
    # If min(sdfs_1_to_2) > 0 and min(sdfs_2_to_1) > 0, they are disjoint.
    # The actual distance is min(min(sdfs_1_to_2), min(sdfs_2_to_1)) if disjoint.
    # If they overlap, the penetration is max(0, -min(sdfs_1_to_2), -min(sdfs_2_to_1)).
    
    # To be safe and smooth, we define the penalty as the square of the maximum violation.
    # Violation = max(0, -min(sdfs_1_to_2), -min(sdfs_2_to_1))
    # But this is not smooth at 0. We can use a smooth max or just rely on L-BFGS-B's ability to handle slight non-smoothness if we use a soft plus.
    
    # Actually, a better metric for "distance" that is differentiable:
    # d = min(sdfs_1_to_2, sdfs_2_to_1)
    # If d < 0, they overlap.
    # Penalty = (min(0, d))^2
    
    min_sdf = np.min(np.concatenate([sdfs_1_to_2, sdfs_2_to_1]))
    if min_sdf < 0:
        return min_sdf**2
    return 0.0

def _objective(params, outer_side, outer_angle):
    """
    Objective function: Minimize the sum of collision penalties + container violation penalties.
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
    """
    # Unpack
    centers = params[0:2*N].reshape(N, 2)
    angles = params[2*N:3*N]
    
    penalty = 0.0
    
    # 1. Container Penalty
    # For each hexagon, check if any vertex is outside the outer hexagon.
    # Outer hexagon is centered at (0,0) with side 'outer_side' and angle 'outer_angle'.
    # We want to penalize vertices that are outside.
    # SDF of outer hexagon: We can reuse _sdf_hex_point but with inverted sign?
    # _sdf_hex_point returns negative if inside.
    # So if SDF > 0, it's outside.
    
    for i in range(N):
        v_i = _hex_vertices(centers[i], 1.0, angles[i])
        for p in v_i:
            # Compute SDF of p to the OUTER hexagon.
            # Note: _sdf_hex_point computes distance to a hexagon of side 1.
            # We need to scale it or compute directly.
            # Let's compute SDF to outer hexagon directly.
            # Outer hexagon side = outer_side.
            # Transform p to local frame of outer hexagon.
            theta = np.deg2rad(outer_angle)
            R = np.array([
                [np.cos(-theta), -np.sin(-theta)],
                [np.sin(-theta), np.cos(-theta)]
            ])
            p_local = R @ p
            apothem = outer_side * np.sqrt(3) / 2.0
            normals = np.array([
                [1.0, 0.0],
                [0.5, np.sqrt(3)/2],
                [-0.5, np.sqrt(3)/2]
            ])
            dists = np.abs(np.dot(p_local, normals.T)) - apothem
            sdf_outer = np.max(dists)
            
            if sdf_outer > 0:
                # Penalty for being outside
                penalty += sdf_outer**2
    
    # 2. Collision Penalty
    for i in range(N):
        for j in range(i + 1, N):
            c1 = centers[i]
            c2 = centers[j]
            a1 = angles[i]
            a2 = angles[j]
            penalty += _sdf_hex_hex(c1, a1, c2, a2)
            
    return penalty

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """
    Strict validity check.
    Returns True if valid, False otherwise.
    """
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check containment
    for i in range(N):
        for p in inner_verts[i]:
            inside = True
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k+1)%6]
                cross_ab = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                if cross_ab < -eps:
                    inside = False
                    break
            if not inside:
                return False

    # Check disjointness
    # Use segment intersection or point-in-polygon.
    # For simplicity and speed in final check, use SAT (Separating Axis Theorem) for convex polygons.
    def _project_polygon(vertices, axis):
        dots = np.dot(vertices, axis)
        return np.min(dots), np.max(dots)

    def _sat_overlap(v1, v2, eps=1e-9):
        axes = []
        for i in range(6):
            edge = v1[(i+1)%6] - v1[i]
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
            edge = v2[(i+1)%6] - v2[i]
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
            
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            min1, max1 = _project_polygon(v1, axis)
            min2, max2 = _project_polygon(v2, axis)
            if max1 < min2 + eps or max2 < min1 + eps:
                return False
        return True

    for i in range(N):
        for j in range(i+1, N):
            if _sat_overlap(inner_verts[i], inner_verts[j]):
                return False
                
    return True

def _compute_required_side(centers, angles, outer_angle=0.0):
    """
    Compute the minimum outer side length required to contain all hexagons.
    """
    max_proj = 0.0
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            verts = _hex_vertices(centers[i], 1.0, angles[i])
            for v in verts:
                proj = np.dot(v, normal)
                if proj > max_proj:
                    max_proj = proj
                    
    return max_proj * 2.0 / np.sqrt(3.0)

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    np.random.seed(42)
    
    # Initial configurations
    initial_configs = []
    
    # 1. Hexagonal lattice (tight packing)
    pts = []
    # Center
    pts.append([0, 0])
    # Ring 1
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)])
    # Ring 2 (partial)
    for k in range(6):
        angle = np.pi/3 * k + np.pi/6
        pts.append([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle)])
        if len(pts) == 12:
            break
    initial_configs.append((np.array(pts), np.zeros(N)))
    
    # 2. Random
    for _ in range(5):
        pts = np.random.uniform(-5, 5, (N, 2))
        initial_configs.append((pts, np.zeros(N)))
        
    # 3. Linear
    pts = np.array([[i*2.0, 0.0] for i in range(N)])
    initial_configs.append((pts, np.zeros(N)))
    
    # Optimization loop
    for outer_angle in [0.0, 30.0]:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        for init_pos, init_ang in initial_configs:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            try:
                # Start with a large container
                current_side = 10.0
                
                pos = init_pos.copy()
                ang = init_ang.copy()
                
                # Iterative shrinking
                for step in range(30):
                    if time.time() - start_time > TIME_LIMIT - 5:
                        break
                        
                    # Compute required side for current config
                    req_side = _compute_required_side(pos, ang, outer_angle)
                    
                    # If current side is larger than required, we can shrink.
                    # We want to optimize the position to fit in a smaller box.
                    # Let's set the target side to req_side * 0.98 (try to shrink by 2%)
                    # Or just use req_side as the container size for the optimizer.
                    # The optimizer will try to minimize the penalty (violations) for this container size.
                    # If it succeeds (penalty ~ 0), we can try a smaller size.
                    
                    target_side = req_side * 0.95
                    
                    # Flatten params
                    params = np.concatenate([pos.flatten(), ang])
                    
                    # Bounds
                    bounds = []
                    for i in range(N):
                        bounds.append((-10, 10)) # x
                        bounds.append((-10, 10)) # y
                        bounds.append((-180, 180)) # angle
                    
                    # Optimize
                    res = minimize(
                        _objective, 
                        params, 
                        args=(target_side, outer_angle),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    
                    if res.success or res.fun < 1e-6:
                        # Check if it's valid
                        new_pos = res.x[0:2*N].reshape(N, 2)
                        new_ang = res.x[2*N:3*N]
                        
                        # Compute new required side
                        new_req_side = _compute_required_side(new_pos, new_ang, outer_angle)
                        
                        # Verify validity
                        inner_hexes = [[new_pos[i][0], new_pos[i][1], new_ang[i]] for i in range(N)]
                        is_valid = _check_validity(inner_hexes, new_req_side, (0.0, 0.0), outer_angle)
                        
                        if is_valid and new_req_side < best_side:
                            best_side = new_req_side
                            best_inner = inner_hexes
                            best_outer_center = (0.0, 0.0)
                            best_outer_angle = outer_angle
                            
                        # Update current config to the optimized one
                        pos = new_pos
                        ang = new_ang
                        
                        # If we couldn't shrink much, stop
                        if new_req_side > target_side * 1.02:
                            break
                    else:
                        # Optimization failed, maybe stuck
                        # Try a slightly larger target? Or break?
                        # Let's break to save time
                        break
                        
            except Exception as e:
                pass
                
    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
