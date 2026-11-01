"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices as (6,2) array."""
    angles = angle_rad + np.pi/3.0 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def compute_required_side(inner, outer_center, outer_angle_deg):
    """Exact minimum outer side length."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    oa = math.radians(outer_angle_deg)
    normals = np.column_stack([np.cos(oa + np.radians(30 + 60*k)), np.sin(oa + np.radians(30 + 60*k))] for k in range(6))
    max_proj = 0.0
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        rel = verts - np.array([ox, oy])
        projs = rel @ normals
        m = projs.max()
        if m > max_proj:
            max_proj = m
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=100.0):
    """Smooth approximation via log-sum-exp over per-hex max projections."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    oa = math.radians(outer_angle_deg)
    normals = np.column_stack([np.cos(oa + np.radians(30 + 60*k)), np.sin(oa + np.radians(30 + 60*k))] for k in range(6))
    
    max_projs = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        rel = verts - np.array([ox, oy])
        projs = rel @ normals  # (6,6)
        per_normal_max = projs.max(axis=0)  # (6,)
        max_projs.append(per_normal_max.max())
    
    if not max_projs:
        return 0.0
    arr = np.array(max_projs)
    m = arr.max()
    if m < -1000:
        return 0.0
    smooth_max = m + math.log(np.sum(np.exp(beta * (arr - m)))) / beta
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (6,2) arrays."""
    axes = []
    for v in (v1, v2):
        for i in range(6):
            edge = v[(i+1)%6] - v[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    for axis in axes:
        p1 = v1 @ axis
        p2 = v2 @ axis
        if p1.max() < p2.min() + eps or p2.max() < p1.min() + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    n = len(inner)
    verts = [hex_vertices(ix, iy, 1.0, math.radians(iang)) for (ix, iy, iang) in inner]
    for i in range(n):
        for j in range(i+1, n):
            if sat_overlap(verts[i], verts[j], eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    ox, oy = outer_center
    outer_v = hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        for v in verts:
            inside = True
            for i in range(6):
                p1 = outer_v[i]
                p2 = outer_v[(i+1)%6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_deg):
    if not check_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def objective(params, use_smooth=True, beta=100.0):
    """Outer side + distance-based overlap penalty."""
    n = N
    inner = []
    for i in range(n):
        cx, cy, ang = params[3*i], params[3*i+1], np.degrees(params[3*i+2]) % 360
        inner.append((cx, cy, ang))
    ox, oy = params[3*n], params[3*n+1]
    outer_angle = np.degrees(params[3*n+2]) % 360
    
    if use_smooth:
        S = compute_required_side_smooth(inner, (ox, oy), outer_angle, beta)
    else:
        S = compute_required_side(inner, (ox, oy), outer_angle)
    
    penalty = 0.0
    min_d = SQRT3
    for i in range(n):
        for j in range(i+1, n):
            d = math.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
            if d < min_d:
                penalty += (min_d - d)**2 * 50.0
    return S + penalty

def axial_to_cart(q, r, scale=1.0):
    x = 1.5 * scale * q
    y = SQRT3 * scale * (r + q/2.0)
    return (x, y)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    axial_coords = [
        (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,-1), (1,1), (-2,1), (-1,-1)
    ][:N]
    
    best_result = None
    best_score = float('inf')
    
    # Generate diverse initial conditions
    ics = []
    
    # IC1: Standard lattice
    s1 = 1.0 + 1e-6
    c1 = [axial_to_cart(q, r, s1) for q, r in axial_coords]
    ic = np.zeros(3*N+3)
    for i, (px, py) in enumerate(c1):
        ic[3*i], ic[3*i+1] = px, py
    ics.append(ic)
    
    # IC2: 30-deg outer
    ic2 = ic.copy(); ic2[3*N+2] = np.radians(30)
    ics.append(ic2)
    
    # IC3: Tighter
    s3 = 0.985
    c3 = [axial_to_cart(q, r, s3) for q, r in axial_coords]
    ic3 = np.zeros(3*N+3)
    for i, (px, py) in enumerate(c3):
        ic3[3*i], ic3[3*i+1] = px, py
    ics.append(ic3)
    
    # IC4: Perturbed
    np.random.seed(42)
    ics.append(ic + np.random.randn(3*N+3)*0.05)
    
    # IC5: 15-deg outer
    ic5 = ic.copy(); ic5[3*N+2] = np.radians(15)
    ics.append(ic5)
    
    # IC6: 30-deg inner + outer
    ic6 = ic.copy()
    for i in range(N):
        ic6[3*i+2] = np.radians(30)
    ic6[3*N+2] = np.radians(30)
    ics.append(ic6)
    
    # IC7: Vertically compressed
    ic7 = ic.copy()
    for i in range(N):
        ic7[3*i+1] *= 0.95
    ics.append(ic7)
    
    # IC8: Asymmetric ring-2 (shift 4 ring-2 hexes)
    ic8 = np.zeros(3*N+3)
    centers8 = [axial_to_cart(0,0,1.0)]
    for k in range(6):
        a = np.radians(60*k)
        centers8.append((SQRT3*np.cos(a), SQRT3*np.sin(a)))
    # 4 ring-2 hexes, slightly pulled inward
    for k in range(4):
        a = np.radians(60*k + 30)
        r = 2*SQRT3*0.96
        centers8.append((r*np.cos(a), r*np.sin(a)))
    for i, (px, py) in enumerate(centers8[:N]):
        ic8[3*i], ic8[3*i+1] = px, py
    ics.append(ic8)
    
    for ic in ics:
        if time.time() > deadline - 25:
            break
        try:
            # Phase 1: smooth, moderate beta
            r1 = minimize(lambda p: objective(p, True, 50.0), ic, method='L-BFGS-B',
                         options={'maxiter':120, 'ftol':1e-10, 'gtol':1e-8})
            # Phase 2: higher beta
            r2 = minimize(lambda p: objective(p, True, 200.0), r1.x, method='L-BFGS-B',
                         options={'maxiter':80, 'ftol':1e-12, 'gtol':1e-10})
            # Phase 3: exact
            r3 = minimize(lambda p: objective(p, False), r2.x, method='L-BFGS-B',
                         options={'maxiter':80, 'ftol':1e-12, 'gtol':1e-10})
            
            best_r = min([r1, r2, r3], key=lambda r: r.fun)
            if best_r.fun < best_score:
                p = best_r.x
                inner = [(p[3*i], p[3*i+1], np.degrees(p[3*i+2])%360) for i in range(N)]
                ox, oy = p[3*N], p[3*N+1]
                oa = np.degrees(p[3*N+2])%360
                S = compute_required_side(inner, (ox, oy), oa)
                if check_validity(inner, (ox, oy), S, oa):
                    best_score = S
                    best_result = (inner, (ox, oy), S, oa)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        cf = [axial_to_cart(q, r, 1.0) for q, r in axial_coords]
        inner = [(px, py, 0.0) for px, py in cf]
        S = compute_required_side(inner, (0,0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    # Refinement pass
    if time.time() < deadline - 20:
        try:
            pb = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                pb[3*i], pb[3*i+1], pb[3*i+2] = cx, cy, np.radians(ang)
            pb[3*N], pb[3*N+1], pb[3*N+2] = best_result[1][0], best_result[1][1], np.radians(best_result[3])
            
            rf = minimize(lambda p: objective(p, False), pb, method='L-BFGS-B',
                         options={'maxiter':150, 'ftol':1e-14, 'gtol':1e-12})
            if rf.fun < best_score:
                p = rf.x
                inner = [(p[3*i], p[3*i+1], np.degrees(p[3*i+2])%360) for i in range(N)]
                ox, oy = p[3*N], p[3*N+1]
                oa = np.degrees(p[3*N+2])%360
                S = compute_required_side(inner, (ox, oy), oa)
                if check_validity(inner, (ox, oy), S, oa):
                    best_result = (inner, (ox, oy), S, oa)
                    best_score = S
        except Exception:
            pass
    
    # Nelder-Mead polish
    if time.time() < deadline - 8 and best_result is not None:
        try:
            pb = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                pb[3*i], pb[3*i+1], pb[3*i+2] = cx, cy, np.radians(ang)
            pb[3*N], pb[3*N+1], pb[3*N+2] = best_result[1][0], best_result[1][1], np.radians(best_result[3])
            
            rn = minimize(lambda p: objective(p, False), pb, method='Nelder-Mead',
                         options={'maxiter':400, 'xatol':1e-10, 'fatol':1e-12})
            if rn.fun < best_score:
                p = rn.x
                inner = [(p[3*i], p[3*i+1], np.degrees(p[3*i+2])%360) for i in range(N)]
                ox, oy = p[3*N], p[3*N+1]
                oa = np.degrees(p[3*N+2])%360
                S = compute_required_side(inner, (ox, oy), oa)
                if check_validity(inner, (ox, oy), S, oa):
                    best_result = (inner, (ox, oy), S, oa)
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
