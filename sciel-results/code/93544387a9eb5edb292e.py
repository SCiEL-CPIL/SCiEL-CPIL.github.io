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

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    def _seg_seg_dist(p1, p2, p3, p4):
        d1 = p2 - p1
        d2 = p4 - p3
        r = p1 - p3
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        
        if a < 1e-12 and e < 1e-12:
            return np.linalg.norm(r)
        elif a < 1e-12:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
            return np.linalg.norm(r + t * d2)
        elif e < 1e-12:
            s = -np.dot(d1, r) / a
            s = max(0.0, min(1.0, s))
            return np.linalg.norm(s * d1 - r)
        
        c = np.dot(d1, r)
        b = np.dot(d1, d2)
        denom = a * e - b * b
        
        if denom < 1e-12:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
            s = max(0.0, min(1.0, (b * t - c) / a))
        else:
            s = (b * f - c * e) / denom
            if s < 0.0 or s > 1.0:
                s = 0.0
                t = f / e
                t = max(0.0, min(1.0, t))
            else:
                t = (b * s + f) / e
                t = max(0.0, min(1.0, t))
                if t < 0.0:
                    t = 0.0
                    s = max(0.0, min(1.0, -c / a))
                elif t > 1.0:
                    t = 1.0
                    s = max(0.0, min(1.0, (b - c) / a))
        
        return np.linalg.norm(s * d1 - r - t * d2)

    def _polygons_overlap(v1, v2, eps=1e-9):
        n1, n2 = len(v1), len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i + 1) % n1]
            for j in range(n2):
                q1 = v2[j]
                q2 = v2[(j + 1) % n2]
                if _seg_seg_dist(p1, p2, q1, q2) < eps:
                    return True
        return False

    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _polygons_overlap(inner_verts[i], inner_verts[j], eps=1e-9):
                return False, "overlap"
            
            is_inside = True
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
                    is_inside = False
                    break
            
            if not is_inside:
                return False, "outside"
                
    return True, "valid"

def _sdf_hexagon_point(point, center, angle_deg, side=1.0):
    """
    Compute signed distance from a point to a regular hexagon.
    Negative if inside, positive if outside.
    Uses the fact that a regular hexagon is the intersection of 6 half-planes.
    """
    theta = np.deg2rad(angle_deg)
    apothem = side * np.sqrt(3) / 2
    
    # For each of the 6 sides, compute the signed distance
    # The hexagon is the intersection of half-planes: dot(p - c, n_k) <= apothem
    # where n_k are the outward normals
    
    # Actually, for a regular hexagon, the distance to the boundary can be computed
    # by finding the minimum over the 6 sides of the distance to each side,
    # but only if the point is in the correct region.
    
    # Simpler approach: use the support function.
    # The hexagon is defined by: max_k dot(p - c, n_k) <= apothem
    # where n_k are the 6 unit normals pointing outward.
    
    # The signed distance (approximately) is: max_k (dot(p-c, n_k) - apothem)
    # But this is only exact for points in the "normal fan" of each face.
    # For a better approximation, we can use this as a convex distance.
    
    cx, cy = center
    px, py = point
    dx = px - cx
    dy = py - cy
    
    max_val = -np.inf
    for k in range(6):
        angle = theta + k * np.pi / 3
        nx = np.cos(angle)
        ny = np.sin(angle)
        val = dx * nx + dy * ny - apothem
        if val > max_val:
            max_val = val
    
    return max_val

def _sdf_hexagon_hexagon(center1, angle1, center2, angle2, side=1.0):
    """
    Compute the minimum signed distance between two regular hexagons.
    Negative if they overlap, positive if they are disjoint.
    We sample points on the boundary of each hexagon and compute distances.
    """
    # For efficiency, we check if any vertex of hex1 is inside hex2 or vice versa,
    # and also check edge-edge distances.
    # But for a smooth penalty, we use a simpler approach:
    # The minimum distance between two convex shapes is the minimum over
    # all pairs of boundary points. We approximate by sampling.
    
    # Actually, for two regular hexagons, we can use the fact that the minimum
    # distance occurs at a vertex of one to an edge of the other, or vertex-vertex.
    
    # For a smooth penalty, let's use a different approach:
    # Compute the Minkowski difference and check if the origin is inside.
    # This is complex. Instead, let's use a simpler smooth penalty based on
    # the distance between centers and the orientations.
    
    # For two unit hexagons, the minimum distance between centers to be disjoint
    # depends on their relative orientation. For the same orientation, it's sqrt(3).
    # For different orientations, it can be less.
    
    # Let's use a conservative estimate: the minimum possible distance is 2*apothem = sqrt(3)
    # for any orientation, but this is not tight.
    
    # Better: compute the actual minimum distance by sampling.
    # Sample 24 points on each hexagon boundary.
    
    verts1 = _hex_vertices(center1, side, angle1)
    verts2 = _hex_vertices(center2, side, angle2)
    
    # For each edge of hex1, compute distance to each edge of hex2
    min_dist = np.inf
    
    for i in range(6):
        p1 = verts1[i]
        p2 = verts1[(i+1) % 6]
        for j in range(6):
            q1 = verts2[j]
            q2 = verts2[(j+1) % 6]
            
            # Distance between two segments
            # Use the segment-segment distance formula
            d1 = p2 - p1
            d2 = q2 - q1
            r = p1 - q1
            a = np.dot(d1, d1)
            e = np.dot(d2, d2)
            f = np.dot(d2, r)
            
            if a < 1e-12 and e < 1e-12:
                dist = np.linalg.norm(r)
            elif a < 1e-12:
                t = f / e
                t = max(0.0, min(1.0, t))
                dist = np.linalg.norm(r + t * d2)
            elif e < 1e-12:
                s = -np.dot(d1, r) / a
                s = max(0.0, min(1.0, s))
                dist = np.linalg.norm(s * d1 - r)
            else:
                c = np.dot(d1, r)
                b = np.dot(d1, d2)
                denom = a * e - b * b
                if denom < 1e-12:
                    s = 0.0
                    t = f / e
                    t = max(0.0, min(1.0, t))
                    s = max(0.0, min(1.0, (b * t - c) / a))
                else:
                    s = (b * f - c * e) / denom
                    if s < 0.0 or s > 1.0:
                        s = 0.0
                        t = f / e
                        t = max(0.0, min(1.0, t))
                    else:
                        t = (b * s + f) / e
                        t = max(0.0, min(1.0, t))
                        if t < 0.0:
                            t = 0.0
                            s = max(0.0, min(1.0, -c / a))
                        elif t > 1.0:
                            t = 1.0
                            s = max(0.0, min(1.0, (b - c) / a))
                
                dist = np.linalg.norm(s * d1 - r - t * d2)
            
            if dist < min_dist:
                min_dist = dist
    
    return min_dist

def _objective_sdf(x, outer_angle, beta=100.0):
    """
    Objective function using SDF-based collision penalty.
    x: 3N vector of (x, y, angle) for each inner hexagon.
    Returns the required outer side length + smooth collision penalty.
    """
    centers = x[:2*N].reshape(N, 2)
    angles = x[2*N:]
    
    # Compute the required outer side length
    # For each inner hexagon, compute its vertices
    # The outer hexagon must contain all vertices
    # The required side length is determined by the maximum projection onto the 6 normals
    
    max_proj = 0.0
    best_i = -1
    best_k = -1
    best_v = None
    best_normal = None
    
    for i in range(N):
        c = centers[i]
        a = angles[i]
        verts = _hex_vertices(c, 1.0, a)
        
        for k in range(6):
            angle = np.deg2rad(outer_angle + k * 60)
            normal = np.array([np.cos(angle), np.sin(angle)])
            for v_idx, v in enumerate(verts):
                proj = np.dot(v, normal)
                if proj > max_proj:
                    max_proj = proj
                    best_i = i
                    best_k = k
                    best_v = v_idx
                    best_normal = normal
    
    side = max_proj * 2.0 / np.sqrt(3.0)
    
    # Compute smooth collision penalty using SDF
    # We want to penalize overlaps. Use a smooth approximation of the max(0, -dist)
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist = _sdf_hexagon_hexagon(centers[i], angles[i], centers[j], angles[j])
            # If dist < 0, they overlap. Penalty = max(0, -dist)^2
            # Smooth approximation: use softplus or just clip
            if dist < 0:
                penalty += (-dist) ** 2
    
    value = side + 10.0 * penalty
    
    # Gradient
    grad = np.zeros(3 * N)
    
    # Gradient of side: (2/sqrt(3)) * normal, applied to the center of the active hexagon
    if best_normal is not None:
        grad[2*best_i] = (2.0 / np.sqrt(3.0)) * best_normal[0]
        grad[2*best_i + 1] = (2.0 / np.sqrt(3.0)) * best_normal[1]
    
    # Gradient of penalty: numerical approximation
    # For simplicity, use finite differences for the penalty part
    # This is expensive but we only have 12 hexagons, so 66 pairs
    # Let's use a simpler approach: just use the side gradient and hope L-BFGS-B
    # can handle the non-smoothness of the penalty.
    
    # Actually, let's compute the gradient of the penalty numerically
    # for the active pairs (those with dist < 0.1)
    for i in range(N):
        for j in range(i + 1, N):
            dist = _sdf_hexagon_hexagon(centers[i], angles[i], centers[j], angles[j])
            if dist < 0.1:
                # Numerical gradient for centers[i], centers[j], angles[i], angles[j]
                eps = 1e-5
                for dim in range(3):
                    for hex_idx in [i, j]:
                        x_plus = x.copy()
                        x_minus = x.copy()
                        offset = 2 * hex_idx + dim
                        x_plus[offset] += eps
                        x_minus[offset] -= eps
                        
                        # Recompute penalty for this pair only
                        def _pair_penalty(x_vec):
                            c1 = x_vec[:2*N].reshape(N, 2)
                            a1 = x_vec[2*N:]
                            d = _sdf_hexagon_hexagon(c1[i], a1[i], c1[j], a1[j])
                            if d < 0:
                                return (-d) ** 2
                            return 0.0
                        
                        f_plus = _pair_penalty(x_plus)
                        f_minus = _pair_penalty(x_minus)
                        grad[offset] += 10.0 * (f_plus - f_minus) / (2 * eps)
    
    return value, grad

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Generate initial guesses
    initial_guesses = []
    
    # Guess 1: Triangular lattice cluster
    lattice_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            lattice_pts.append([x, y])
    lattice_pts = np.array(lattice_pts)
    
    picked = []
    sorted_idx = np.argsort(np.sum(lattice_pts**2, axis=1))
    for idx in sorted_idx:
        p = lattice_pts[idx]
        valid = True
        for q in picked:
            if np.linalg.norm(p - q) < np.sqrt(3) * 0.99:
                valid = False
                break
        if valid:
            picked.append(p)
            if len(picked) == 12:
                break
    if len(picked) == 12:
        centers = np.array(picked)
        angles = np.zeros(N)
        initial_guesses.append(np.concatenate([centers.flatten(), angles]))
    
    # Guess 2: Random
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.uniform(-3, 3, (N, 2))
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(pts[i] - pts[j]) < np.sqrt(3):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            angles = np.random.uniform(0, 360, N)
            initial_guesses.append(np.concatenate([pts.flatten(), angles]))
    
    # Guess 3: Hexagonal arrangement
    pts = [[0,0]]
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)])
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle)])
        if len(pts) == 12:
            break
    centers = np.array(pts)
    angles = np.zeros(N)
    initial_guesses.append(np.concatenate([centers.flatten(), angles]))
    
    # Optimize
    for guess in initial_guesses:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        for outer_angle in [0.0, 30.0]:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            try:
                res = minimize(
                    _objective_sdf,
                    guess,
                    args=(outer_angle, 100.0),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                x_opt = res.x
                centers_opt = x_opt[:2*N].reshape(N, 2)
                angles_opt = x_opt[2*N:]
                
                # Compute the actual side length
                max_proj = 0.0
                for i in range(N):
                    c = centers_opt[i]
                    a = angles_opt[i]
                    verts = _hex_vertices(c, 1.0, a)
                    for k in range(6):
                        angle = np.deg2rad(outer_angle + k * 60)
                        normal = np.array([np.cos(angle), np.sin(angle)])
                        for v in verts:
                            proj = np.dot(v, normal)
                            if proj > max_proj:
                                max_proj = proj
                
                side = max_proj * 2.0 / np.sqrt(3.0)
                
                # Check validity
                inner_hexes = [[centers_opt[i][0], centers_opt[i][1], angles_opt[i]] for i in range(N)]
                
                is_valid, reason = _check_validity(inner_hexes, side, (0.0, 0.0), outer_angle)
                
                if is_valid and side < best_side:
                    best_side = side
                    best_inner = inner_hexes
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = outer_angle
                    
            except Exception as e:
                pass
    
    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
