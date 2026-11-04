import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import itertools

N = 12
TIME_LIMIT = 145.0 # Safety margin for 150s limit

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
    
    # Check disjointness via segment intersection
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
            
            # Check containment
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

def _get_cluster_offsets():
    """
    Returns the local coordinates of the 4 hexagons in a compact 2x2 cluster.
    Hexagons are unit size, angle 0.
    To fit 4 hexagons tightly:
    Hex 0: (0, 0)
    Hex 1: (sqrt(3), 0)  -> Side by side
    Hex 2: (sqrt(3)/2, 1.5) -> Top neighbor of 0? 
    Distance between (0,0) and (sqrt(3)/2, 1.5) is sqrt(3/4 + 2.25) = sqrt(3) = 1.732. Correct.
    Hex 3: (3*sqrt(3)/2, 1.5) -> Top neighbor of 1.
    
    Let's verify non-overlap.
    0 and 1: dist sqrt(3). OK.
    0 and 2: dist sqrt(3). OK.
    0 and 3: dist sqrt((1.5*sqrt(3))^2 + 1.5^2) = sqrt(6.75 + 2.25) = 3. OK.
    1 and 2: dist sqrt((0.5*sqrt(3))^2 + 1.5^2) = sqrt(0.75 + 2.25) = sqrt(3). OK.
    1 and 3: dist sqrt(3). OK.
    2 and 3: dist sqrt(3). OK.
    
    So offsets are:
    [[0, 0],
     [sqrt(3), 0],
     [sqrt(3)/2, 1.5],
     [3*sqrt(3)/2, 1.5]]
    """
    s3 = np.sqrt(3)
    return np.array([
        [0.0, 0.0],
        [s3, 0.0],
        [s3/2.0, 1.5],
        [3*s3/2.0, 1.5]
    ])

def _compute_bounding_side(centers, angles, outer_angle=0.0):
    """
    Compute the minimum side length of the outer hexagon (centered at 0,0) 
    containing the given hexagons.
    centers: (N, 2) array
    angles: (N,) array of degrees
    """
    max_proj = 0.0
    # Outer hexagon normals
    # If outer angle is 0, vertices are at 0, 60, ... degrees.
    # Normals are at 30, 90, ... degrees? 
    # Let's stick to the definition:
    # A regular hexagon with side S and rotation alpha has vertices at R_k.
    # The half-width (apothem) is S * sqrt(3) / 2.
    # The condition for a point P to be inside is dot(P, n_k) <= apothem for all 6 normals n_k.
    # The normals of a hexagon rotated by alpha are at angles alpha + 30 + 60*k.
    # Wait, if vertices are at alpha + 60*k, the normals (perpendicular to edges) are at alpha + 30 + 60*k.
    # Let's use this.
    
    base_normal_angles = np.deg2rad(outer_angle + 30 + np.arange(6)*60)
    normals = np.array([np.cos(base_normal_angles), np.sin(base_normal_angles)]).T
    
    for i in range(len(centers)):
        c = centers[i]
        a = angles[i]
        # Vertices of inner hexagon i
        theta = np.deg2rad(a)
        verts = c + np.array([np.cos(theta + np.arange(6)*np.pi/3), 
                               np.sin(theta + np.arange(6)*np.pi/3)]).T
        
        # Project all vertices onto all normals
        # Projections: (6, 6) matrix
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
        
    return max_proj * 2.0 / np.sqrt(3.0)

def _cluster_objective(params, cluster_offsets, outer_angle=0.0):
    """
    params: 9 floats [x0, y0, a0, x1, y1, a1, x2, y2, a2]
    cluster_offsets: (4, 2) local coords
    Returns side length
    """
    centers = np.zeros((12, 2))
    angles = np.zeros(12)
    
    for cluster_idx in range(3):
        x = params[cluster_idx*3]
        y = params[cluster_idx*3 + 1]
        a_deg = params[cluster_idx*3 + 2]
        
        # Rotate offsets
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # Apply rotation to offsets
        # offset_x' = x*cos - y*sin
        # offset_y' = x*sin + y*cos
        ox = cluster_offsets[:, 0]
        oy = cluster_offsets[:, 1]
        
        rox = ox * cos_a - oy * sin_a
        roy = ox * sin_a + oy * cos_a
        
        for k in range(4):
            idx = cluster_idx * 4 + k
            centers[idx] = [x + rox[k], y + roy[k]]
            angles[idx] = a_deg
            
    return _compute_bounding_side(centers, angles, outer_angle)

def _check_cluster_collision(params, cluster_offsets, eps=1e-6):
    """
    Check if any two hexagons from different clusters overlap.
    Returns True if valid (no overlap).
    """
    centers = np.zeros((12, 2))
    angles = np.zeros(12)
    
    for cluster_idx in range(3):
        x = params[cluster_idx*3]
        y = params[cluster_idx*3 + 1]
        a_deg = params[cluster_idx*3 + 2]
        
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        ox = cluster_offsets[:, 0]
        oy = cluster_offsets[:, 1]
        
        rox = ox * cos_a - oy * sin_a
        roy = ox * sin_a + oy * cos_a
        
        for k in range(4):
            idx = cluster_idx * 4 + k
            centers[idx] = [x + rox[k], y + roy[k]]
            angles[idx] = a_deg
            
    # Check all pairs
    for i in range(12):
        for j in range(i+1, 12):
            # Skip if same cluster (they are fixed valid internally)
            if i // 4 == j // 4:
                continue
                
            v1 = _hex_vertices(centers[i], 1.0, angles[i])
            v2 = _hex_vertices(centers[j], 1.0, angles[j])
            
            # Quick bounding box check
            if (np.max(v1[:,0]) < np.min(v2[:,0]) - eps or 
                np.max(v2[:,0]) < np.min(v1[:,0]) - eps or
                np.max(v1[:,1]) < np.min(v2[:,1]) - eps or 
                np.max(v2[:,1]) < np.min(v1[:,1]) - eps):
                continue
                
            # Segment intersection check
            # This is expensive, but we only do it for close pairs
            # Let's use a simpler distance check first? 
            # Actually, segment intersection is robust.
            
            # To speed up, we can check distance between centers.
            # Min distance between centers for non-overlap is sqrt(3) if aligned.
            # But with rotation, it can be smaller? No, the inradius is sqrt(3)/2.
            # So if dist < sqrt(3), they might overlap.
            # If dist > 2*sqrt(3)/2 * 2 = 2*sqrt(3) ~ 3.46, they definitely don't.
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0 * np.sqrt(3):
                continue
            
            # Full check
            for k in range(6):
                p1 = v1[k]
                p2 = v1[(k+1)%6]
                for m in range(6):
                    q1 = v2[m]
                    q2 = v2[(m+1)%6]
                    # Segment distance
                    # Reusing logic from _check_validity but inline for speed
                    d1 = p2 - p1
                    d2 = q2 - q1
                    r = p1 - q1
                    a = np.dot(d1, d1)
                    e = np.dot(d2, d2)
                    f = np.dot(d2, r)
                    
                    if a < 1e-12 and e < 1e-12:
                        if np.linalg.norm(r) < eps: return False
                    elif a < 1e-12:
                        t = f / e
                        t = max(0.0, min(1.0, t))
                        if np.linalg.norm(r + t * d2) < eps: return False
                    elif e < 1e-12:
                        s = -np.dot(d1, r) / a
                        s = max(0.0, min(1.0, s))
                        if np.linalg.norm(s * d1 - r) < eps: return False
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
                        
                        if np.linalg.norm(s * d1 - r - t * d2) < eps:
                            return False
    return True

def construct():
    start_time = time.time()
    cluster_offsets = _get_cluster_offsets()
    
    # Bounds for the 9 parameters
    # x, y in [-5, 5], angle in [0, 60)
    bounds = [(-5, 5), (-5, 5), (0, 60)] * 3
    
    # Differential Evolution
    # We want to minimize the side length.
    # But we must ensure no collisions.
    # DE doesn't handle constraints well directly.
    # We will use a penalty function.
    
    def _de_objective(params):
        side = _cluster_objective(params, cluster_offsets, 0.0)
        if not _check_cluster_collision(params, cluster_offsets):
            return side + 100.0
        return side
        
    # Run DE
    # Maxiter limited to keep time down
    max_iter = 200
    pop_size = 20
    
    try:
        res = differential_evolution(
            _de_objective, 
            bounds, 
            maxiter=max_iter, 
            popsize=pop_size,
            seed=42,
            tol=1e-6,
            polish=True # Use L-BFGS-B for polish? Might be slow or fail. Let's try False first and polish manually.
        )
        
        best_params = res.x
        best_side = _cluster_objective(best_params, cluster_offsets, 0.0)
        
        # If DE found a valid solution, refine it
        if _check_cluster_collision(best_params, cluster_offsets):
            # Local refinement using L-BFGS-B
            # We need gradients.
            # Side length is a max of linear functions.
            # We can approximate gradient or use finite differences.
            # Given the low dimension (9), finite differences are cheap enough.
            
            def _refine_obj(p):
                s = _cluster_objective(p, cluster_offsets, 0.0)
                if not _check_cluster_collision(p, cluster_offsets):
                    return s + 10.0 * (s - best_side) # Penalty if it breaks
                return s
                
            # Use Nelder-Mead for robustness in 9D with non-smooth penalty
            res_refine = minimize(
                _refine_obj, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res_refine.fun < best_side:
                best_params = res_refine.x
                best_side = res_refine.fun
                
    except Exception:
        pass
        
    # If we didn't find a good solution or time is running out, try a few random restarts with local search
    # Or just use the best from DE.
    
    # Let's try a few more random starts if we have time
    current_best_side = float('inf')
    best_final_params = None
    
    # Check the DE result
    if best_params is not None and _check_cluster_collision(best_params, cluster_offsets):
        current_best_side = best_side
        best_final_params = best_params

    # Random restarts
    np.random.seed(42)
    time_remaining = TIME_LIMIT - (time.time() - start_time)
    attempts = 0
    max_attempts = 50
    
    while time_remaining > 10.0 and attempts < max_attempts:
        # Generate random params
        p = np.random.uniform(bounds[0][0], bounds[0][1], 3)
        p[2] = np.random.uniform(0, 60)
        p2 = np.random.uniform(bounds[0][0], bounds[0][1], 3)
        p2[2] = np.random.uniform(0, 60)
        p3 = np.random.uniform(bounds[0][0], bounds[0][1], 3)
        p3[2] = np.random.uniform(0, 60)
        params = np.concatenate([p, p2, p3])
        
        # Quick check
        if not _check_cluster_collision(params, cluster_offsets):
            attempts += 1
            continue
            
        side = _cluster_objective(params, cluster_offsets, 0.0)
        
        if side < current_best_side:
            current_best_side = side
            best_final_params = params.copy()
            
            # Try to refine this local optimum
            def _refine_obj_local(p):
                s = _cluster_objective(p, cluster_offsets, 0.0)
                if not _check_cluster_collision(p, cluster_offsets):
                    return s + 10.0
                return s
                
            try:
                res_refine = minimize(
                    _refine_obj_local, 
                    params, 
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6}
                )
                if res_refine.fun < current_best_side and _check_cluster_collision(res_refine.x, cluster_offsets):
                    current_best_side = res_refine.fun
                    best_final_params = res_refine.x
            except:
                pass
                
        attempts += 1
        time_remaining = TIME_LIMIT - (time.time() - start_time)

    if best_final_params is None:
        # Fallback: return a very large hexagon
        return [[10.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    # Construct the final inner hexagons
    centers = np.zeros((12, 2))
    angles = np.zeros(12)
    
    for cluster_idx in range(3):
        x = best_final_params[cluster_idx*3]
        y = best_final_params[cluster_idx*3 + 1]
        a_deg = best_final_params[cluster_idx*3 + 2]
        
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        ox = cluster_offsets[:, 0]
        oy = cluster_offsets[:, 1]
        
        rox = ox * cos_a - oy * sin_a
        roy = ox * sin_a + oy * cos_a
        
        for k in range(4):
            idx = cluster_idx * 4 + k
            centers[idx] = [x + rox[k], y + roy[k]]
            angles[idx] = a_deg
            
    # Calculate final side
    final_side = _compute_bounding_side(centers, angles, 0.0)
    
    # Verify
    inner_hexes = [[centers[i][0], centers[i][1], angles[i]] for i in range(12)]
    is_valid, reason = _check_validity(inner_hexes, final_side, (0.0, 0.0), 0.0)
    
    if not is_valid:
        # Fallback
        return [[10.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    return (inner_hexes, (0.0, 0.0), final_side, 0.0)

