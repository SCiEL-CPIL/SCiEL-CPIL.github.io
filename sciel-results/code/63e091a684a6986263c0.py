import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    """
    Returns squared distance from point (px, py) to the hexagon if outside,
    or 0 if inside.
    """
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -a to align hexagon with axes
    rx = dx * math.cos(-a) - dy * math.sin(-a)
    ry = dx * math.sin(-a) + dy * math.cos(-a)
    
    apothem = s * math.sqrt(3) / 2
    sum_sq_viol = 0.0
    for i in range(3):
        theta_n = math.radians(30 + i * 60)
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if proj > apothem:
            sum_sq_viol += (proj - apothem)**2
        if proj < -apothem:
            sum_sq_viol += (-apothem - proj)**2
    return sum_sq_viol

def _pairwise_penalty(p1, p2):
    """
    p1, p2: (cx, cy, angle_deg)
    Returns penalty if they overlap.
    """
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    dx = c1x - c2x
    dy = c1y - c2y
    if dx*dx + dy*dy > 4.0:
        return 0.0
        
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    penalty = 0.0
    
    for pt in v1:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c2x, c2y, 1.0, a2)
        
    for pt in v2:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c1x, c1y, 1.0, a1)
        
    if penalty == 0.0:
        # Check edge intersections
        def segments_intersect(p1, p2, p3, p4):
            def cross(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False

        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    mid1 = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                    mid2 = ((p3[0]+p4[0])/2, (p3[1]+p4[1])/2)
                    penalty += _point_in_hex_dist_sq(mid1[0], mid1[1], c2x, c2y, 1.0, a2)
                    penalty += _point_in_hex_dist_sq(mid2[0], mid2[1], c1x, c1y, 1.0, a1)
                    if penalty == 0.0:
                        penalty += 1.0
                
    return penalty

def _outer_penalty(inner_params, outer_params):
    """
    inner_params: list of (cx, cy, angle)
    outer_params: (cx, cy, s, angle)
    Returns penalty if any inner hexagon is not fully inside the outer hexagon.
    """
    ocx, ocy, os, oang = outer_params
    
    penalty = 0.0
    for ip in inner_params:
        icx, icy, iang = ip
        v_in = _hex_vertices(icx, icy, 1.0, iang)
        for pt in v_in:
            penalty += _point_in_hex_dist_sq(pt[0], pt[1], ocx, ocy, os, oang)
            
    return penalty

def _objective(x):
    """
    x: [cx1, cy1, a1, ..., cxN, cyN, aN, ocx, ocy, os, oang]
    """
    n_inner = N
    inner_params = []
    for i in range(n_inner):
        cx = x[i*3]
        cy = x[i*3+1]
        a = x[i*3+2]
        inner_params.append((cx, cy, a))
        
    ocx = x[n_inner*3]
    ocy = x[n_inner*3+1]
    os = x[n_inner*3+2]
    oang = x[n_inner*3+3]
    outer_params = (ocx, ocy, os, oang)
    
    pen_pair = 0.0
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            pen_pair += _pairwise_penalty(inner_params[i], inner_params[j])
            
    pen_outer = _outer_penalty(inner_params, outer_params)
    
    return os + 100.0 * (pen_pair + pen_outer)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial guess: Lattice packing
    init_inner = []
    init_inner.append((0.0, 0.0, 0.0))
    for i in range(6):
        a = math.radians(60*i)
        x = 1.5 * math.cos(a)
        y = 1.5 * math.sin(a)
        init_inner.append((x, y, 0.0))
    # Ring 2 (5 more to make 11)
    # We need 4 more. Let's pick 4 from ring 2.
    # Ring 2 coords: (3,0), (2.25, 1.9), (0.75, 2.6), (-0.75, 2.6), (-2.25, 1.9), (-3, 0), (-2.25, -1.9), (-0.75, -2.6), (0.75, -2.6), (2.25, -1.9)
    # Let's add (3,0), (-3,0), (0, 2.6), (0, -2.6) roughly?
    # Actually, let's just use a simple grid for init to be safe.
    # The DE will move them.
    # Let's add 4 more around.
    init_inner.append((3.0, 0.0, 0.0))
    init_inner.append((-3.0, 0.0, 0.0))
    init_inner.append((0.0, 3.0, 0.0))
    init_inner.append((0.0, -3.0, 0.0))
    
    x0 = []
    for p in init_inner:
        x0.append(p[0])
        x0.append(p[1])
        x0.append(p[2])
        
    x0.extend([0.0, 0.0, 5.0, 0.0])
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 360))
        
    bounds.append((-10, 10))
    bounds.append((-10, 10))
    bounds.append((3.0, 10.0))
    bounds.append((0, 360))
    
    best_x = None
    best_val = float('inf')
    
    try:
        res = differential_evolution(_objective, bounds, maxiter=20, popsize=5, seed=42, polish=False)
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
        
    if time.time() > deadline:
        if best_x is not None:
            return _format_result(best_x)
        else:
            return _format_result(x0)
            
    if best_x is not None:
        try:
            res2 = minimize(_objective, best_x, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            if res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
        except Exception:
            pass
    
    return _format_result(best_x if best_x is not None else x0)

def _format_result(x):
    inner = []
    for i in range(N):
        cx = x[i*3]
        cy = x[i*3+1]
        a = x[i*3+2]
        inner.append([cx, cy, a])
        
    ocx = x[N*3]
    ocy = x[N*3+1]
    os = x[N*3+2]
    oang = x[N*3+3]
    
    return inner, (ocx, ocy), os, oang

if __name__ == "__main__":
    r = construct()
    print(r)
