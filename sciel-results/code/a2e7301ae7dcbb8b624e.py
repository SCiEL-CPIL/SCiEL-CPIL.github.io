import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 145.0

def _hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit hexagon centered at (cx,cy) rotated by angle_deg."""
    a = np.deg2rad(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 (pointy-top)
    base = np.array([[np.cos(np.deg2rad(k*60)), np.sin(np.deg2rad(k*60))] for k in range(6)])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return base @ R.T + np.array([cx, cy])

def _point_in_polygon(px, py, poly):
    """Ray casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _seg_seg_dist(p1, p2, p3, p4):
    """Min distance between two line segments."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(r)
    elif a < 1e-12:
        t2 = np.clip(f / e, 0, 1)
        return np.linalg.norm(r + d2 * t2)
    elif e < 1e-12:
        t1 = np.clip(-np.dot(d1, r) / a, 0, 1)
        return np.linalg.norm(r + d1 * t1)
    else:
        c = np.dot(d1, r)
        b = np.dot(d1, d2)
        denom = a * e - b * b
        t1 = np.clip((b * f - c * e) / denom, 0, 1)
        t2 = np.clip((a * f - b * c) / denom, 0, 1)
        if t2 < 1e-9:
            t1 = np.clip(-np.dot(d1, r) / a, 0, 1)
            t2 = 0
        else:
            t2 = np.clip((a * f - b * c) / denom, 0, 1)
        return np.linalg.norm(r + d1 * t1 - d2 * t2)

def _poly_separation(poly1, poly2):
    """Min distance between two convex polygons (negative if overlapping)."""
    n1, n2 = len(poly1), len(poly2)
    min_dist = float('inf')
    # Check if any vertex of poly1 is inside poly2 or vice versa
    for i in range(n1):
        if _point_in_polygon(poly1[i][0], poly1[i][1], poly2):
            return -1.0
    for i in range(n2):
        if _point_in_polygon(poly2[i][0], poly2[i][1], poly1):
            return -1.0
    # Check all edge pairs
    for i in range(n1):
        p1 = poly1[i]
        p2 = poly1[(i+1) % n1]
        for j in range(n2):
            p3 = poly2[j]
            p4 = poly2[(j+1) % n2]
            d = _seg_seg_dist(p1, p2, p3, p4)
            if d < min_dist:
                min_dist = d
    return min_dist

def _fast_side_and_penalty(x, beta=50.0):
    """
    Fast evaluation: x = [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10]
    Returns (side_length, penalty) where penalty is smooth approximation of max(0, -min_sep).
    """
    n = len(x) // 3
    centers = x[0:n*3:3].copy()
    ys = x[1:n*3:3].copy()
    angles = x[2:n*3:3].copy()
    
    # Compute all vertices
    all_verts = []
    for i in range(n):
        verts = _hex_vertices(centers[i], ys[i], angles[i])
        all_verts.append(verts)
    
    # Find max extent for each of 6 hexagon orientations
    # For a regular hexagon of side S with angle theta, the support function in direction d is:
    # h(d) = S * max_k dot(d, v_k) where v_k are unit hex vertices
    # We need: for all directions d, max_i (dot(d, center_i) + h_inner(d)) <= dot(d, outer_center) + S * h_outer(d)
    # For centered outer hex: S * max_k dot(d, w_k) >= max_i (dot(d, center_i) + max_j dot(d, v_ij))
    
    # Simplify: try 12 directions (6 for hex sides, 6 for vertices)
    best_side = 0.0
    for theta_deg in [0, 30, 60, 90, 120, 150]:
        theta = np.deg2rad(theta_deg)
        # Outer hex vertices (unit)
        outer_v = np.array([[np.cos(theta + np.deg2rad(k*60)), np.sin(theta + np.deg2rad(k*60))] for k in range(6)])
        
        # For each direction (normal to each side of outer hex), compute required S
        for k in range(6):
            # Normal direction (outward)
            mid_angle = theta + np.deg2rad(30 + k*60)
            d = np.array([np.cos(mid_angle), np.sin(mid_angle)])
            
            # Support of outer hex in direction d: S * dot(d, w) where w is vertex
            # Actually for regular hex, h(d) = S * max_j dot(d, v_j)
            h_outer_unit = np.max(outer_v @ d)
            
            # Support of inner cluster in direction d
            h_inner = 0.0
            for i in range(n):
                # Inner hex i support in direction d
                h_i = np.max(all_verts[i] @ d) - centers[i]*d[0] - ys[i]*d[1]
                # Wait, all_verts[i] already includes center, so:
                h_i = np.max(all_verts[i] @ d)
                h_inner = max(h_inner, h_i)
            
            # Constraint: S * h_outer_unit >= h_inner
            if h_outer_unit > 1e-9:
                S_req = h_inner / h_outer_unit
            else:
                S_req = 0
            best_side = max(best_side, S_req)
    
    # Also check vertex directions of outer hex
    for theta_deg in [0, 30, 60, 90, 120, 150]:
        theta = np.deg2rad(theta_deg)
        outer_v = np.array([[np.cos(theta + np.deg2rad(k*60)), np.sin(theta + np.deg2rad(k*60))] for k in range(6)])
        for k in range(6):
            d = outer_v[k]
            d = d / np.linalg.norm(d)
            h_outer_unit = np.max(outer_v @ d)
            h_inner = 0.0
            for i in range(n):
                h_i = np.max(all_verts[i] @ d)
                h_inner = max(h_inner, h_i)
            if h_outer_unit > 1e-9:
                S_req = h_inner / h_outer_unit
                best_side = max(best_side, S_req)
    
    # Overlap penalty: sum of soft penalties for each pair
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            sep = _poly_separation(all_verts[i], all_verts[j])
            if sep < 0.01:
                # Smooth penalty: exp(-beta * sep) when sep < 0
                penalty += np.exp(-beta * sep)
    
    return best_side, penalty

def _objective(x, beta=50.0):
    side, pen = _fast_side_and_penalty(x, beta)
    return side + 1000.0 * pen

def _is_valid(x, tol=1e-6):
    """Check if configuration is valid (no overlaps)."""
    n = len(x) // 3
    all_verts = []
    for i in range(n):
        verts = _hex_vertices(x[i*3], x[i*3+1], x[i*3+2])
        all_verts.append(verts)
    for i in range(n):
        for j in range(i+1, n):
            sep = _poly_separation(all_verts[i], all_verts[j])
            if sep < -tol:
                return False
    return True

def _seed_solution():
    """Return the seed flower arrangement as x vector."""
    centers = [(0.0, 0.0)]
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        centers.append((r * np.cos(angle), r * np.sin(angle)))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0
        centers.append((r * np.cos(angle), r * np.sin(angle)))
    
    x = np.zeros(3 * N)
    for i, (cx, cy) in enumerate(centers):
        x[i*3] = cx
        x[i*3+1] = cy
        x[i*3+2] = 30.0
    return x

def construct():
    t0 = time.time()
    
    # Start from seed
    x0 = _seed_solution()
    best_x = x0.copy()
    best_side, best_pen = _fast_side_and_penalty(x0)
    
    # Try L-BFGS-B optimization
    try:
        res = minimize(_objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        if time.time() - t0 < DEADLINE - 10:
            x_opt = res.x
            side_opt, pen_opt = _fast_side_and_penalty(x_opt)
            if pen_opt < 1e-3 and side_opt < best_side:
                best_x = x_opt
                best_side = side_opt
                best_pen = pen_opt
    except Exception:
        pass
    
    # Try Nelder-Mead as backup
    if time.time() - t0 < DEADLINE - 15:
        try:
            res2 = minimize(_objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
            if time.time() - t0 < DEADLINE - 5:
                x_opt2 = res2.x
                side_opt2, pen_opt2 = _fast_side_and_penalty(x_opt2)
                if pen_opt2 < 1e-3 and side_opt2 < best_side:
                    best_x = x_opt2
                    best_side = side_opt2
                    best_pen = pen_opt2
        except Exception:
            pass
    
    # Extract best solution
    n = N
    inner = []
    for i in range(n):
        inner.append([best_x[i*3], best_x[i*3+1], best_x[i*3+2]])
    
    outer_center = (0.0, 0.0)
    outer_angle = 30.0
    
    # Add tiny margin
    side = best_side + 1e-9
    
    return inner, outer_center, side, outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
