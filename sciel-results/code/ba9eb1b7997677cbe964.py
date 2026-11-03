"""Pack 12 unit hexagons in minimal outer hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_verts(cx, cy, s, angle_rad):
    """Vertices of regular hexagon (flat-top), center (cx,cy), side s, rotation angle_rad."""
    verts = []
    for k in range(6):
        theta = angle_rad + np.radians(60 * k)
        x = cx + s * np.cos(theta)
        y = cy + s * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _point_in_convex(px, py, verts):
    """Check if point is inside convex polygon (CCW vertices)."""
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < -1e-9:
            return False
    return True

def _point_seg_dist(px, py, ax, ay, bx, by):
    """Distance from point to line segment."""
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return np.sqrt((px - ax)**2 + (py - ay)**2)
    t = ((px - ax) * dx + (py - ay) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return np.sqrt((px - cx)**2 + (py - cy)**2)

def _seg_seg_dist(p1, p2, p3, p4):
    """Min distance between two segments."""
    d1x = p2[0] - p1[0]; d1y = p2[1] - p1[1]
    d2x = p4[0] - p3[0]; d2y = p4[1] - p3[1]
    rx = p1[0] - p3[0]; ry = p1[1] - p3[1]
    a = d1x*d1x + d1y*d1y
    e = d2x*d2x + d2y*d2y
    f = d2x*rx + d2y*ry
    if a < 1e-12 and e < 1e-12:
        return np.sqrt(rx*rx + ry*ry)
    if a < 1e-12:
        t2 = max(0.0, min(1.0, f/e))
        dx = rx + t2*d2x; dy = ry + t2*d2y
        return np.sqrt(dx*dx + dy*dy)
    c = d1x*rx + d1y*ry
    if e < 1e-12:
        t1 = max(0.0, min(1.0, -c/a))
        dx = rx + t1*d1x; dy = ry + t1*d1y
        return np.sqrt(dx*dx + dy*dy)
    b = d1x*d2x + d1y*d2y
    denom = a*e - b*b
    if denom < 1e-12:
        t1 = max(0.0, min(1.0, -c/a))
        dx = rx + t1*d1x; dy = ry + t1*d1y
        return np.sqrt(dx*dx + dy*dy)
    t1 = (b*f - c*e) / denom
    t1 = max(0.0, min(1.0, t1))
    t2 = (b*t1 + f) / e
    if t2 < 0:
        t1 = max(0.0, min(1.0, -c/a))
        t2 = 0
    elif t2 > 1:
        t1 = max(0.0, min(1.0, (b-c)/a))
        t2 = 1
    dx = rx + t1*d1x - t2*d2x
    dy = ry + t1*d1y - t2*d2y
    return np.sqrt(dx*dx + dy*dy)

def _poly_min_dist(v1, v2):
    """Min distance between two convex polygons. 0 if overlapping."""
    # Check vertex containment
    for v in v1:
        if _point_in_convex(v[0], v[1], v2):
            return 0.0
    for v in v2:
        if _point_in_convex(v[0], v[1], v1):
            return 0.0
    # Edge-edge
    n1, n2 = len(v1), len(v2)
    min_d = np.inf
    for i in range(n1):
        p1 = v1[i]; p2 = v1[(i+1)%n1]
        for j in range(n2):
            p3 = v2[j]; p4 = v2[(j+1)%n2]
            d = _seg_seg_dist(p1, p2, p3, p4)
            if d < min_d:
                min_d = d
    return min_d

def _inner_in_outer_margin(inner_v, outer_v):
    """Min distance from inner polygon to outer boundary. Negative if any vertex outside."""
    min_margin = np.inf
    for v in inner_v:
        if not _point_in_convex(v[0], v[1], outer_v):
            # Compute distance to outer boundary
            min_d = np.inf
            n = len(outer_v)
            for k in range(n):
                d = _point_seg_dist(v[0], v[1], outer_v[k][0], outer_v[k][1], 
                                     outer_v[(k+1)%n][0], outer_v[(k+1)%n][1])
                min_d = min(min_d, d)
            min_margin = min(min_margin, -min_d)
        else:
            # Compute distance to boundary (positive margin)
            min_d = np.inf
            n = len(outer_v)
            for k in range(n):
                d = _point_seg_dist(v[0], v[1], outer_v[k][0], outer_v[k][1],
                                     outer_v[(k+1)%n][0], outer_v[(k+1)%n][1])
                min_d = min(min_d, d)
            min_margin = min(min_margin, min_d)
    return min_margin

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    # Hex lattice initial positions
    def hex_to_cart(q, r):
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        return x, y
    
    # 12 hexagons: center + ring1(6) + 5 from ring2
    axial = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
             (2,0),(1,1),(0,2),(-1,2),(-2,2)]
    
    init_pos = []
    for q,r in axial:
        x,y = hex_to_cart(q,r)
        init_pos.append([x, y, 0.0])
    
    max_r = max(np.sqrt(p[0]**2+p[1]**2) for p in init_pos)
    init_s = max_r + 1.5
    
    params0 = []
    for p in init_pos:
        params0.extend(p)
    params0.extend([0.0, 0.0, init_s, 0.0])
    params0 = np.array(params0)
    
    def objective(p):
        ip = p[:3*N].reshape(N,3)
        ocx, ocy, os, oa = p[3*N], p[3*N+1], p[3*N+2], p[3*N+3]
        os = max(os, 0.5)
        ov = _hex_verts(ocx, ocy, os, oa)
        pen = 0.0
        ivs = []
        for i in range(N):
            x,y,a = ip[i]
            iv = _hex_verts(x,y,1.0,a)
            ivs.append(iv)
            m = _inner_in_outer_margin(iv, ov)
            if m < 0:
                pen += m*m*500
        for i in range(N):
            for j in range(i+1,N):
                d = _poly_min_dist(ivs[i], ivs[j])
                if d < 1e-4:
                    pen += (1e-4-d)*5000
        return os + pen
    
    best_p = params0.copy()
    best_f = objective(params0)
    
    for attempt in range(3):
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(objective, best_p, method='Nelder-Mead',
                          options={'maxiter':3000,'xatol':1e-5,'fatol':1e-5,'adaptive':True})
            if res.fun < best_f:
                best_f = res.fun
                best_p = res.x.copy()
        except:
            pass
    
    ip = best_p[:3*N].reshape(N,3)
    ocx, ocy, os, oa = best_p[3*N], best_p[3*N+1], best_p[3*N+2], best_p[3*N+3]
    os = max(os, 3.0)
    
    inner = [[float(ip[i][0]), float(ip[i][1]), float(np.degrees(ip[i][2]))] for i in range(N)]
    return inner, (float(ocx), float(ocy)), float(os), float(np.degrees(oa))
