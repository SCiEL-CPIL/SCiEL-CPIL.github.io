import numpy as np
import time
from itertools import combinations

N = 12
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin for 120s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments."""
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

def _min_dist_polygons(v1, v2):
    """Calculate minimum distance between two convex polygons."""
    min_d = float('inf')
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        for j in range(len(v2)):
            q1 = v2[j]
            q2 = v2[(j + 1) % len(v2)]
            d = _seg_seg_dist(p1, p2, q1, q2)
            if d < min_d:
                min_d = d
    return min_d

def _point_in_hexagon(p, center, side, angle_deg, eps=1e-9):
    """Check if point p is inside the hexagon defined by center, side, angle."""
    verts = _hex_vertices(center, side, angle_deg)
    inside = True
    for i in range(6):
        a = verts[i]
        b = verts[(i + 1) % 6]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cross < -eps:
            inside = False
            break
    return inside

def _objective(params):
    """
    Objective function for optimization.
    params: 
    - 24 floats: 12 inner hexagon centers (x, y)
    - 12 floats: 12 inner hexagon angles (degrees)
    - 2 floats: outer hexagon center (x, y)
    - 1 float: outer hexagon side length
    - 1 float: outer hexagon angle (degrees)
    Total: 40 floats.
    
    Returns:
    - Side length if valid
    - Side length + penalty if invalid
    """
    # Unpack
    inner_centers = np.array(params[0:24]).reshape(12, 2)
    inner_angles = np.array(params[24:36])
    outer_center = np.array(params[36:38])
    outer_side = params[38]
    outer_angle = params[39]
    
    # Ensure side is positive
    if outer_side < 0:
        outer_side = 1e-6
    
    # Compute vertices
    inner_verts = []
    for i in range(12):
        v = _hex_vertices(inner_centers[i], 1.0, inner_angles[i])
        inner_verts.append(v)
        
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    
    penalty = 0.0
    
    # Check disjointness
    for i in range(12):
        for j in range(i + 1, 12):
            d = _min_dist_polygons(inner_verts[i], inner_verts[j])
            if d < 1e-5:
                # Penalty for overlap: how much they overlap (approx by -d if d is negative? 
                # _min_dist_polygons returns distance between segments, which is 0 if they touch/overlap.
                # We need a measure of overlap. 
                # If they overlap, we can check if any vertex of one is inside the other.
                # Simple penalty: if distance is 0, add a large penalty.
                # Better: check if they intersect properly.
                # For now, if d < 1e-5, assume overlap and add penalty based on how deep the overlap is.
                # A common trick: penalty = max(0, -d) doesn't work if d is always >= 0.
                # Let's use a penalty that increases if they are "too close" (d < 0 is impossible for distance).
                # So if d < 1e-5, we are in contact or overlapping.
                # We want to push them apart.
                # If they overlap, the "distance" is 0. We need to penalize the overlap area or depth.
                # Approximation: If they overlap, penalty = 1.0. This is binary.
                # To make it continuous, we can check if the center of one is inside the other.
                c_i = inner_centers[i]
                c_j = inner_centers[j]
                if _point_in_hexagon(c_i, inner_centers[j], 1.0, inner_angles[j], eps=1e-9):
                    penalty += 1.0
                if _point_in_hexagon(c_j, inner_centers[i], 1.0, inner_angles[i], eps=1e-9):
                    penalty += 1.0
                # If neither center is inside, they might still overlap (e.g. corners).
                # But for convex polygons, if they overlap, at least one vertex of one is inside the other?
                # No, but if they overlap, the distance is 0.
                # Let's add a small penalty for being close to encourage separation.
                penalty += max(0, 1e-5 - d) * 100000 # Strong penalty for touching
    
    # Check containment
    for i in range(12):
        for p in inner_verts[i]:
            if not _point_in_hexagon(p, outer_center, outer_side, outer_angle, eps=1e-9):
                # Point is outside. Penalty proportional to distance outside.
                # Calculate distance to outer hexagon boundary.
                # Approximation: distance to the nearest edge.
                min_d = float('inf')
                for k in range(6):
                    a = outer_verts[k]
                    b = outer_verts[(k + 1) % 6]
                    d = _seg_seg_dist(p, p, a, b) # Point to segment distance
                    if d < min_d:
                        min_d = d
                # If p is outside, min_d is the distance to the boundary.
                # We want to push it in. So penalty is min_d.
                penalty += min_d * 100000
                
    return outer_side + penalty

def construct():
    start_time = time.time()
    
    # Initial guess: Hexagonal packing
    # Center at (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 5 hexagons (to make 12)
    
    # Triangular lattice basis
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2.0])
    
    # Lattice points
    points = []
    # Center
    points.append(np.array([0.0, 0.0]))
    
    # Ring 1
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0: continue
            if abs(i) + abs(j) + abs(i+j) <= 2: # Hex distance 1
                p = i * v1 + j * v2
                # Only take 6 of them? No, ring 1 has 6 points.
                # Hex distance 1 points:
                # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
                # Let's just generate all and pick.
                pass
    
    # Simpler initial guess:
    # 12 points in a 3x4 grid? No, hexagonal.
    # Let's use a known good starting config:
    # 3 rows of 4?
    # Or just a cluster.
    
    # Let's generate 12 points from a hexagonal lattice centered at 0.
    # We want them to be disjoint.
    # Min distance between centers should be > sqrt(3) ~ 1.732.
    # Lattice distance 1 is 1.0. Too close.
    # Lattice distance 2 is 2.0. OK.
    
    # So we pick 12 points from the lattice such that no two are at distance 1.
    # This is an independent set.
    
    # Candidates within radius 3
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            if np.linalg.norm(p) <= 3.0:
                candidates.append(p)
    
    # Greedy pick 12
    picked = []
    for p in candidates:
        if len(picked) >= 12: break
        valid = True
        for q in picked:
            if np.linalg.norm(p - q) < 1.8:
                valid = False
                break
        if valid:
            picked.append(p)
            
    if len(picked) < 12:
        # Fallback: just place them in a line
        picked = [np.array([i, 0]) for i in range(-5, 7)]
        picked = picked[:12]
        
    # Initial params
    inner_centers = np.array(picked)
    inner_angles = np.zeros(12)
    # Outer center: centroid
    outer_center = np.mean(inner_centers, axis=0)
    # Outer side: estimate
    max_r = np.max(np.linalg.norm(inner_centers - outer_center, axis=1))
    outer_side = max_r * 2 / np.sqrt(3) + 2.0
    outer_angle = 0.0
    
    params0 = np.concatenate([inner_centers.flatten(), inner_angles, [outer_center[0], outer_center[1]], [outer_side], [outer_angle]])
    
    # Optimization
    # Use Nelder-Mead
    # Max time 110s
    deadline = start_time + 110.0
    
    # We need a time-aware optimizer.
    # scipy.optimize.minimize doesn't have a time limit.
    # We can run it in a loop or just set maxiter high and hope.
    # Better: implement a simple loop with time check.
    
    # Let's use scipy.optimize.minimize with maxiter and check time after.
    # Or just run it and interrupt if it takes too long?
    # The problem is small (40 vars). NM should be fast.
    
    from scipy.optimize import minimize
    
    best_result = None
    best_side = float('inf')
    
    # Try a few restarts with different initial angles/positions
    np.random.seed(42)
    
    for restart in range(3):
        if time.time() > deadline:
            break
            
        # Perturb initial guess
        noise = np.random.normal(0, 0.1, len(params0))
        # Keep angles small
        noise[24:36] = np.random.normal(0, 10, 12)
        params = params0 + noise
        
        try:
            res = minimize(_objective, params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.success or res.fun < best_side:
                # Verify validity of the result
                p = res.x
                inner_centers = np.array(p[0:24]).reshape(12, 2)
                inner_angles = np.array(p[24:36])
                outer_center = np.array(p[36:38])
                outer_side = p[38]
                outer_angle = p[39]
                
                # Check if penalty is 0 (valid)
                # Re-evaluate objective to see if penalty is 0
                # If penalty > 0, it's invalid.
                # We can check _objective - outer_side. If close to 0, it's valid.
                val = _objective(p)
                penalty = val - outer_side
                
                if penalty < 1e-3: # Valid
                    if outer_side < best_side:
                        best_side = outer_side
                        best_result = (
                            [[inner_centers[i][0], inner_centers[i][1], inner_angles[i]] for i in range(12)],
                            [outer_center[0], outer_center[1]],
                            outer_side,
                            outer_angle
                        )
        except Exception:
            pass
            
    if best_result is not None:
        return best_result
        
    # Fallback: Return the seed configuration if search fails
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

