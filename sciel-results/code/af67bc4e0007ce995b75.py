import numpy as np
import time
import random
import math

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
    Compute the signed distance from point p to a regular hexagon.
    Positive if outside, negative if inside.
    """
    # Rotate point to hexagon's local frame
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    x = cos_t * dx + sin_t * dy
    y = -sin_t * dx + cos_t * dy
    
    # Hexagon is defined by |x| <= s, |y| <= s*sqrt(3)/2, |x|+|y|/sqrt(3) <= s? 
    # No, standard regular hexagon with flat top/bottom or pointy?
    # _hex_vertices uses angles starting from theta.
    # If theta=0, vertices are at 0, 60, 120...
    # This is a "pointy" hexagon relative to x-axis?
    # Vertices: (s, 0), (s/2, s*sqrt(3)/2)...
    # Let's derive the SDF for this specific orientation.
    
    # The hexagon is the intersection of 6 half-planes.
    # For a regular hexagon of side s, the apothem is a = s * sqrt(3) / 2.
    # The normals to the edges are at angles: 30, 90, 150, 210, 270, 330 (if flat top)
    # Or 0, 60, 120... (if pointy side)?
    
    # Let's check _hex_vertices(0,0,1,0):
    # Angles: 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3
    # v0: (1, 0)
    # v1: (0.5, sqrt(3)/2)
    # Edge v0-v1: vector (-0.5, sqrt(3)/2). Normal: (-sqrt(3)/2, -0.5) ?? 
    # Let's compute normals explicitly.
    # Actually, it's easier to use the property that the distance from center to edge is a = s*sqrt(3)/2.
    # The 6 outward normals for this orientation (vertices at 0, 60, ...) are at angles:
    # 30, 90, 150, 210, 270, 330 degrees.
    # Let's verify. Edge between 0 and 60 deg. Midpoint angle 30 deg. Normal points outward at 30 deg.
    # Yes.
    
    # In local coordinates (rotated by -theta), the normals are at fixed angles:
    # 30, 90, 150, 210, 270, 330.
    # Let's define the 6 normals in local frame.
    angles_normals = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(angles_normals), np.sin(angles_normals)])
    
    # Vector from center to point in local frame
    p_local = np.array([x, y])
    
    # Project p_local onto each normal
    projections = normals @ p_local
    
    # The hexagon is defined by: dot(n_i, p) <= a for all i
    # Where a = side * sqrt(3) / 2
    a = side * np.sqrt(3) / 2.0
    
    # The SDF is the maximum of (dot(n_i, p) - a) over all i?
    # Wait, if p is inside, dot(n_i, p) < a for all i.
    # The distance to the boundary is min_i (a - dot(n_i, p)).
    # If inside, SDF is negative. So SDF = -min_i (a - dot(n_i, p)) = max_i (dot(n_i, p) - a).
    # If outside, this gives the distance to the closest face?
    # For a convex polygon, the SDF is indeed max(0, max_i (dot(n_i, p) - a)) ?
    # No, that's only correct if the point is in the "normal cone" of a face.
    # For a regular hexagon, due to symmetry, this formula holds for the entire space?
    # Let's check. If p is far away in direction 0 deg (x-axis).
    # Normal 30 deg: dot = x * cos(30).
    # Normal 90 deg: dot = y * 1.
    # If y=0, dot(30) = 0.866 x. dot(270) = 0.
    # max is 0.866 x - a.
    # Actual distance to hexagon along x-axis:
    # Vertex is at (s, 0). Distance is x - s.
    # s = 1. a = 0.866.
    # Formula: 0.866 x - 0.866 = 0.866 (x - 1).
    # Actual: x - 1.
    # They differ by a factor!
    # The SDF is not just the max of plane distances if we are outside the Voronoi region of the face.
    
    # Correct SDF for regular polygon:
    # 1. Compute d = max_i (dot(n_i, p) - a).
    # 2. If d <= 0, return d (inside or on boundary).
    # 3. If d > 0, we are outside. The closest feature might be a vertex.
    # We need to find which "sector" the point is in.
    # For a regular hexagon, the vertices are at angles 0, 60, 120... in local frame.
    # The bisectors of the vertices are at 0, 60, 120...
    # If the point's angle in local frame is close to a vertex angle, the distance is to the vertex.
    
    # Let's use a simpler, robust method for SDF of a convex polygon:
    # It's the maximum of the signed distances to the edges, BUT only if the projection of p onto the edge line falls within the edge segment.
    # Otherwise, it's the distance to the vertex.
    
    # Actually, for optimization, a smooth approximation is often sufficient and faster.
    # But let's try to get the exact SDF for better gradients.
    
    # Alternative: Use the fact that the hexagon is the Minkowski sum of a point and 3 line segments? No.
    
    # Let's stick to the "Max of Plane Distances" but corrected for vertices.
    # For a regular hexagon, the SDF is:
    # Let r = norm(p).
    # Let alpha = angle(p).
    # Map alpha to the range [0, 60) by symmetry.
    # In this sector, the closest feature is either the face at 30 deg or the vertex at 0 or 60.
    
    # Let's implement the exact SDF for a regular hexagon in local coords.
    # Symmetry: 6 sectors.
    # Sector 1: angle in [-30, 30] relative to x-axis? No, vertices at 0, 60.
    # Bisector at 0.
    # Let's normalize angle to [0, 2pi).
    # Reduce to [0, pi/3] using symmetry.
    
    ang = np.arctan2(y, x)
    # Normalize to [0, 2pi)
    ang = ang % (2 * np.pi)
    
    # Map to [0, pi/3]
    # If ang in [0, pi/3]: sector 0 (vertex at 0, face at pi/6)
    # If ang in [pi/3, 2pi/3]: sector 1 (vertex at pi/3, face at pi/2)
    # ...
    
    # Let's just use the property:
    # d_face = max_i (dot(n_i, p) - a)
    # If d_face <= 0, return d_face.
    # Else, find the closest vertex.
    
    # Vertices in local frame:
    v_angles = np.arange(6) * (np.pi / 3)
    vertices = np.column_stack([side * np.cos(v_angles), side * np.sin(v_angles)])
    
    # Distances to vertices
    dists_v = np.linalg.norm(p_local[:, None] - vertices[None, :], axis=1)
    min_dist_v = np.min(dists_v)
    
    # Distances to faces (signed)
    # Normals at 30, 90, ...
    n_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
    d_faces = normals @ p_local - a
    
    max_d_face = np.max(d_faces)
    
    # If max_d_face <= 0, we are inside (or on face). SDF = max_d_face.
    # If max_d_face > 0, we are outside.
    # The SDF is min(max_d_face, min_dist_v)?
    # Let's check.
    # If outside, the distance to the polygon is the minimum distance to any boundary element.
    # The boundary consists of edges and vertices.
    # The distance to an edge (segment) is either the perpendicular distance (if projection falls on segment) or distance to endpoint.
    # max_d_face gives the perpendicular distance to the infinite line of the face, but only if we are in the normal cone.
    # If we are in the normal cone of face i, then d_face_i is the distance to the edge.
    # If we are not, the closest point is a vertex.
    
    # So, SDF = min( max_d_face, min_dist_v ) ?
    # Let's test with point (2, 0), side=1.
    # a = 0.866.
    # p_local = (2, 0).
    # d_faces:
    # n0 (30 deg): 2*0.866 - 0.866 = 0.866.
    # n5 (330 deg): 2*0.866 - 0.866 = 0.866.
    # Others negative.
    # max_d_face = 0.866.
    # Vertices: (1,0), (0.5, 0.866)...
    # dist to (1,0) is 1.
    # min_dist_v = 1.
    # min(0.866, 1) = 0.866.
    # Actual distance: 1 (to vertex (1,0)).
    # So min is wrong. It should be 1.
    
    # The condition for using d_face is that the projection of p onto the edge line lies within the edge.
    # For face i with normal n_i, the edge is perpendicular to n_i.
    # The projection of p onto the line is p - dot(p, n_i) n_i.
    # This projection must lie between the two vertices of that edge.
    
    # This is getting complex. Let's use a smooth approximation for the SDF which is differentiable everywhere.
    # SDF_approx = max(0, max_d_face) + smooth_correction?
    # Or just use max_d_face as a penalty. It underestimates the distance when near vertices, but provides a valid gradient direction to push out.
    # For packing, "pushing out" along the normal of the most violated face is usually effective.
    
    # Let's use max_d_face as the SDF for inside/outside check and gradient.
    # It is 0 on the boundary.
    # It is negative inside.
    # It is positive outside.
    # It is a valid "barrier" function.
    
    return max_d_face

def _sdf_hex_hex(c1, a1, c2, a2, side=1.0):
    """
    Approximate SDF between two hexagons.
    We sample points on the boundary of hex1 and check their SDF against hex2, and vice versa.
    To be fast, we only check vertices and edge midpoints.
    """
    v1 = _hex_vertices(c1, side, a1)
    v2 = _hex_vertices(c2, side, a2)
    
    # Check vertices of 1 against 2
    min_d = np.inf
    for p in v1:
        d = _sdf_hex_point(p, c2, side, a2)
        if d < min_d:
            min_d = d
            
    # Check vertices of 2 against 1
    for p in v2:
        d = _sdf_hex_point(p, c1, side, a1)
        if d < min_d:
            min_d = d
            
    # Also check edge midpoints for better accuracy?
    # For 12 hexagons, 66 pairs. 12 vertices per pair. 792 SDF calls per eval.
    # SDF call is cheap (vector ops).
    # Let's add edge midpoints.
    
    mids1 = (v1 + np.roll(v1, -1, axis=0)) / 2.0
    for p in mids1:
        d = _sdf_hex_point(p, c2, side, a2)
        if d < min_d:
            min_d = d
            
    mids2 = (v2 + np.roll(v2, -1, axis=0)) / 2.0
    for p in mids2:
        d = _sdf_hex_point(p, c1, side, a1)
        if d < min_d:
            min_d = d
            
    return min_d

def _objective(params, outer_center, outer_side, outer_angle):
    """
    Objective: Minimize outer_side.
    Penalty for:
    1. Inner hexagons overlapping each other.
    2. Inner hexagons outside outer hexagon.
    
    Returns: outer_side + Penalty
    """
    centers = params[:, :2]
    angles = params[:, 2]
    
    penalty = 0.0
    
    # 1. Inner-Inner Collisions
    for i in range(N):
        for j in range(i + 1, N):
            d = _sdf_hex_hex(centers[i], angles[i], centers[j], angles[j])
            if d < 0:
                # Overlap. Penalty proportional to depth squared.
                penalty += (d * d) * 10.0
                
    # 2. Containment
    # SDF of outer hexagon (inverted: negative inside, positive outside)
    # We want inner hexagons to be inside outer.
    # So SDF_outer(p) <= 0 for all p in inner hex.
    # Penalty if SDF_outer(p) > 0.
    
    for i in range(N):
        v_i = _hex_vertices(centers[i], 1.0, angles[i])
        # Check vertices and midpoints of inner hex
        mids_i = (v_i + np.roll(v_i, -1, axis=0)) / 2.0
        points = np.vstack([v_i, mids_i])
        
        for p in points:
            d = _sdf_hex_point(p, outer_center, outer_side, outer_angle)
            if d > 0:
                penalty += (d * d) * 10.0
                
    return outer_side + penalty

def _compute_min_side(params, outer_center, outer_angle):
    """
    Compute the minimum side length of the outer hexagon to contain all inner hexagons.
    """
    centers = params[:, :2]
    angles = params[:, 2]
    
    max_proj = 0.0
    # Outer hexagon normals
    theta = np.deg2rad(outer_angle)
    n_angles = theta + np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
    
    for i in range(N):
        v_i = _hex_vertices(centers[i], 1.0, angles[i])
        # Project vertices onto normals
        # p - center_outer
        rel = v_i - outer_center
        projs = rel @ normals.T
        # The boundary is at distance a = side * sqrt(3) / 2
        # So we need max_proj <= a
        # side = max_proj * 2 / sqrt(3)
        max_proj = max(max_proj, np.max(projs))
        
    return max_proj * 2.0 / np.sqrt(3.0)

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_params = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try a few outer angles
    outer_angles = [0.0, 30.0]
    
    # Initial Guess: Hexagonal packing
    # Center at 0,0.
    # 6 around it.
    # 5 more?
    # Let's generate a good initial guess.
    init_params = np.zeros((N, 3))
    # Hex 0 at center
    init_params[0] = [0, 0, 0]
    # Hex 1-6 around center
    # Distance between centers of touching hexagons is 2 * apothem = 2 * sqrt(3)/2 * 1 = sqrt(3)?
    # No, side=1. Apothem = sqrt(3)/2.
    # Distance between centers = 2 * apothem = sqrt(3) if flat-to-flat.
    # If pointy-to-pointy?
    # Regular hexagons pack with centers at distance sqrt(3) if aligned.
    d = np.sqrt(3)
    for k in range(6):
        ang = np.pi/3 * k
        init_params[k+1] = [d * np.cos(ang), d * np.sin(ang), 0.0]
        
    # Hex 7-11?
    # Place them further out.
    for k in range(5):
        ang = np.pi/3 * k + np.pi/6 # Offset
        init_params[k+7] = [2*d * np.cos(ang), 2*d * np.sin(ang), 0.0]
        
    initial_guesses = [init_params]
    
    # Add random guesses
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.uniform(-5, 5, (N, 2))
        angs = np.random.choice([0, 30, 60, 90, 120, 150], N)
        p = np.hstack([pts, angs.reshape(-1, 1)])
        initial_guesses.append(p)

    for outer_angle in outer_angles:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        for guess in initial_guesses:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            # Simulated Annealing
            T = 1.0
            cooling_rate = 0.999
            min_T = 1e-6
            
            current_params = guess.copy()
            current_side = _compute_min_side(current_params, (0,0), outer_angle)
            current_val = _objective(current_params, (0,0), current_side, outer_angle)
            
            best_params_sa = current_params.copy()
            best_val_sa = current_val
            
            while T > min_T:
                if time.time() - start_time > TIME_LIMIT - 2:
                    break
                    
                # Perturb
                new_params = current_params.copy()
                # Move one hexagon
                idx = np.random.randint(N)
                new_params[idx, 0] += np.random.normal(0, 0.1)
                new_params[idx, 1] += np.random.normal(0, 0.1)
                new_params[idx, 2] += np.random.choice([-30, -15, 0, 15, 30])
                
                new_side = _compute_min_side(new_params, (0,0), outer_angle)
                new_val = _objective(new_params, (0,0), new_side, outer_angle)
                
                delta = new_val - current_val
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_params = new_params
                    current_side = new_side
                    current_val = new_val
                    
                    if current_val < best_val_sa:
                        best_val_sa = current_val
                        best_params_sa = current_params.copy()
                        
                T *= cooling_rate
                
            # After SA, check validity and update global best
            final_params = best_params_sa
            final_side = _compute_min_side(final_params, (0,0), outer_angle)
            
            # Verify validity
            is_valid = True
            # Check overlaps
            for i in range(N):
                for j in range(i+1, N):
                    d = _sdf_hex_hex(final_params[i,:2], final_params[i,2], final_params[j,:2], final_params[j,2])
                    if d < -1e-5:
                        is_valid = False
                        break
                if not is_valid: break
            
            # Check containment
            if is_valid:
                for i in range(N):
                    v_i = _hex_vertices(final_params[i,:2], 1.0, final_params[i,2])
                    for p in v_i:
                        d = _sdf_hex_point(p, (0,0), final_side, outer_angle)
                        if d > 1e-5:
                            is_valid = False
                            break
                    if not is_valid: break
            
            if is_valid and final_side < best_side:
                best_side = final_side
                best_params = final_params
                best_outer_center = (0.0, 0.0)
                best_outer_angle = outer_angle

    if best_params is not None:
        # Format output
        inner_hexes = [[best_params[i][0], best_params[i][1], best_params[i][2]] for i in range(N)]
        return (inner_hexes, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

