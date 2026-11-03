import numpy as np
import time
import sys

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

def _point_in_poly(p, poly):
    """Check if point p is inside convex polygon poly."""
    n = len(poly)
    signs = []
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
        if cross != 0:
            signs.append(1 if cross > 0 else -1)
    if not signs:
        return True
    return all(s == signs[0] for s in signs)

def _seg_seg_intersect(p1, p2, p3, p4):
    def _orient(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val > 0: return 1
        if val < 0: return -1
        return 0
    o1 = _orient(p1, p2, p3)
    o2 = _orient(p1, p2, p4)
    o3 = _orient(p3, p4, p1)
    o4 = _orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    return False

def _seg_seg_dist(p1, p2, p3, p4):
    """Calculate minimum distance between two line segments."""
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
    """Calculate minimum distance between two convex polygons. Returns 0 if intersecting."""
    n1, n2 = len(v1), len(v2)
    
    # Check vertex containment
    for p in v1:
        if _point_in_poly(p, v2):
            return 0.0
    for p in v2:
        if _point_in_poly(p, v1):
            return 0.0

    # Check edge intersections
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            q1 = v2[j]
            q2 = v2[(j + 1) % n2]
            if _seg_seg_intersect(p1, p2, q1, q2):
                return 0.0

    # If disjoint, calculate min distance between edges
    min_d = float('inf')
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            q1 = v2[j]
            q2 = v2[(j + 1) % n2]
            d = _seg_seg_dist(p1, p2, q1, q2)
            if d < min_d:
                min_d = d
    return min_d

def _outer_side_for_config(centers, outer_angle_deg):
    """
    Calculate the minimum outer side length required to contain all inner hexagons
    with the given centers and outer angle.
    Inner hexagons are fixed at angle 0.
    """
    base_verts = np.array([
        [np.cos(np.pi/3 * k), np.sin(np.pi/3 * k)] for k in range(6)
    ])
    
    theta = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        a = theta + np.pi/3 * k
        normals.append([np.cos(a), np.sin(a)])
    normals = np.array(normals)
    
    max_proj = 0.0
    for c in centers:
        verts = c + base_verts
        projs = verts @ normals.T
        local_max = np.max(projs)
        if local_max > max_proj:
            max_proj = local_max
            
    return max_proj * 2.0 / np.sqrt(3.0)

def _check_validity(centers, outer_side, outer_center, outer_angle_deg, eps=1e-7):
    """
    Check if the configuration is valid:
    1. Inner hexagons are disjoint.
    2. Inner hexagons are inside the outer hexagon.
    """
    base_verts = np.array([
        [np.cos(np.pi/3 * k), np.sin(np.pi/3 * k)] for k in range(6)
    ])
    inner_verts = [c + base_verts for c in centers]
    
    # 1. Disjointness
    for i in range(N):
        for j in range(i + 1, N):
            d = _min_dist_polygons(inner_verts[i], inner_verts[j])
            if d < eps:
                return False

    # 2. Containment
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle_deg)
    
    # Get outward normals of outer hexagon
    theta = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        a = theta + np.pi/3 * k
        normals.append([np.cos(a), np.sin(a)])
    normals = np.array(normals)
    
    apothem = outer_side * np.sqrt(3) / 2
    
    for v in inner_verts:
        dists = v @ normals.T
        if np.any(dists > apothem + eps):
            return False
            
    return True

def construct():
    start_time = time.time()
    
    # Strategy:
    # 1. Initialize with a compact hexagonal cluster (lattice).
    # 2. Use gradient-based optimization to minimize outer side length.
    # 3. Use penalty functions for collisions and containment to guide the gradient.
    # 4. Verify the final result with a strict geometric check.
    
    # Initial configuration: Hexagonal cluster of 12 hexagons
    # Center (0,0), 6 neighbors, 5 more in the second ring to make 12.
    # Actually, 1+6+5=12.
    # Let's generate lattice points and pick the closest 12 to origin.
    pts = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts)
    d = np.sum(pts**2, axis=1)
    idx = np.argsort(d)[:12]
    centers = pts[idx].copy()
    
    # Flatten for optimization: [x1, y1, x2, y2, ..., x12, y12, outer_angle]
    # We will optimize centers and outer angle simultaneously.
    # Outer center is fixed at (0,0) for symmetry, but we can translate centers to center them.
    # Actually, translating all centers doesn't change relative distances, but changes projection on outer.
    # To minimize outer size, we should center the configuration.
    # Let's add a global translation (tx, ty) to the optimization variables.
    # Variables: [cx1, cy1, ..., cx12, cy12, tx, ty, outer_angle_deg]
    # Total: 24 + 2 + 1 = 27 variables.
    
    x0 = np.zeros(27)
    for i in range(12):
        x0[i*2] = centers[i][0]
        x0[i*2+1] = centers[i][1]
    x0[24] = 0.0 # tx
    x0[25] = 0.0 # ty
    x0[26] = 0.0 # outer angle
    
    def objective(x):
        if time.time() - start_time > TIME_LIMIT - 5:
            return 1e6 # Timeout penalty
            
        centers_opt = x[:24].reshape(12, 2)
        tx = x[24]
        ty = x[25]
        outer_angle = x[26]
        
        shifted_centers = centers_opt + np.array([tx, ty])
        
        # 1. Calculate outer side length required for containment
        side = _outer_side_for_config(shifted_centers, outer_angle)
        
        # 2. Collision penalty
        # We want min distance between any pair to be >= 0.
        # Penalty = sum(max(0, eps - dist)^2)
        # To be efficient, we only check pairs that are close.
        # But for gradient, we need smoothness.
        # Let's use a soft penalty: sum(exp(-(dist/eps)^2)) or similar?
        # Simpler: if dist < 0, penalty is large.
        # Since they are convex, if they overlap, dist=0 in our _min_dist_polygons.
        # We need a differentiable proxy.
        # Approximation: Distance between centers minus sum of inradii?
        # Inradius of unit hex is sqrt(3)/2.
        # If center_dist < sqrt(3), they might collide.
        # A better proxy for "collision" in gradient descent is to penalize
        # the square of the overlap depth if we had it, but we don't.
        # Let's use a heuristic penalty based on center distance.
        # If center_dist < 1.0, they definitely collide (vertex radius 1.0, so dist<1.0 means vertices overlap).
        # Actually, min distance between centers for disjoint unit hexagons is 1.0 (touching vertices).
        # So penalty for dist < 1.0.
        
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                d_c = np.linalg.norm(shifted_centers[i] - shifted_centers[j])
                # If d_c < 1.0, they are likely colliding or very close.
                # Use a smooth ramp.
                if d_c < 1.0:
                    penalty += (1.0 - d_c)**2
        
        # 3. Containment penalty
        # If side is calculated by max projection, it's always "contained" by definition of side.
        # So we don't need a containment penalty if we define side as the minimal containing side.
        # The objective is just to minimize `side` + `penalty`.
        
        return side + 10.0 * penalty
    
    # Use Nelder-Mead or BFGS?
    # Nelder-Mead is derivative-free and robust for this small dimension.
    # BFGS is faster if we can provide gradients, but our penalty is not easily differentiable
    # due to the max() and min_dist logic.
    # Let's try Nelder-Mead first. It's slower per step but doesn't need gradients.
    # Or use a simple coordinate descent?
    # Nelder-Mead is a good choice for ~30 dimensions.
    
    from scipy.optimize import minimize
    
    best_x = x0
    best_val = objective(x0)
    
    # Try a few restarts with slight perturbations
    np.random.seed(42)
    for restart in range(5):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        if restart == 0:
            x_start = x0
        else:
            x_start = x0 + np.random.normal(0, 0.1, 27)
        
        try:
            res = minimize(objective, x_start, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Extract best solution
    centers_opt = best_x[:24].reshape(12, 2)
    tx = best_x[24]
    ty = best_x[25]
    outer_angle = best_x[26]
    
    shifted_centers = centers_opt + np.array([tx, ty])
    
    # Calculate final side
    side = _outer_side_for_config(shifted_centers, outer_angle)
    
    # Verify validity
    # Note: The optimization might have left some hexagons slightly overlapping if penalty wasn't strong enough
    # or if the heuristic penalty didn't perfectly match geometric collision.
    # We must verify.
    
    if _check_validity(shifted_centers, side, (0.0, 0.0), outer_angle):
        inner = [[shifted_centers[i][0], shifted_centers[i][1], 0.0] for i in range(N)]
        return (inner, (0.0, 0.0), side, outer_angle)
    else:
        # Fallback: Return the initial lattice configuration which is known to be valid (touching)
        # The initial lattice points are disjoint (touching).
        # Let's re-verify the initial lattice configuration.
        init_centers = centers
        init_side = _outer_side_for_config(init_centers, 0.0)
        if _check_validity(init_centers, init_side, (0.0, 0.0), 0.0):
            inner = [[init_centers[i][0], init_centers[i][1], 0.0] for i in range(N)]
            return (inner, (0.0, 0.0), init_side, 0.0)
        
        # Ultimate fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

# Run construct to ensure it works
# construct()
