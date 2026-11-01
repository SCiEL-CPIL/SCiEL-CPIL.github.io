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
    normals = np.column_stack([
        np.array([np.cos(oa + math.radians(30 + 60*k)),
                  np.sin(oa + math.radians(30 + 60*k))])
        for k in range(6)
    ])
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
    normals = np.column_stack([
        np.array([np.cos(oa + math.radians(30 + 60*k)),
                  np.sin(oa + math.radians(30 + 60*k))])
        for k in range(6)
    ])
    
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

def axial_to_cart(q, r, scale=1.0):
    x = 1.5 * scale * q
    y = SQRT3 * scale * (r + q/2.0)
    return (x, y)

def build_rigid_inner(cx, cy, theta_inner_rad, scale=1.0):
    """Build inner hexagon list from rigid composite parameters."""
    axial_coords = [
        (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,-1), (1,1), (-2,1), (-1,-1)
    ][:N]
    inner = []
    for q, r in axial_coords:
        bx, by = axial_to_cart(q, r, scale)
        # Rotate base position around origin, then translate
        bxr = cx + bx * math.cos(theta_inner_rad) - by * math.sin(theta_inner_rad)
        byr = cy + bx * math.sin(theta_inner_rad) + by * math.cos(theta_inner_rad)
        inner.append((bxr, byr, math.degrees(theta_inner_rad) % 360))
    return inner

def objective_rigid(params, use_smooth=True, beta=100.0, scale=1.0):
    """Objective for rigid composite: 6 DOF (cx, cy, theta_inner, ox, oy, theta_outer)."""
    cx, cy = params[0], params[1]
    theta_inner = params[2]
    ox, oy = params[3], params[4]
    theta_outer = params[5]
    
    inner = build_rigid_inner(cx, cy, theta_inner, scale)
    oa_deg = math.degrees(theta_outer) % 360
    
    if use_smooth:
        S = compute_required_side_smooth(inner, (ox, oy), oa_deg, beta)
    else:
        S = compute_required_side(inner, (ox, oy), oa_deg)
    
    # Overlap penalty (distance-based)
    penalty = 0.0
    min_d = SQRT3 * scale
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            d = math.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
            if d < min_d:
                penalty += (min_d - d)**2 * 50.0
    return S + penalty

def objective_full(params, use_smooth=True, beta=100.0):
    """Objective for full free DOF: 3N+3 params."""
    n = N
    inner = []
    for i in range(n):
        cx, cy, ang = params[3*i], params[3*i+1], math.degrees(params[3*i+2]) % 360
        inner.append((cx, cy, ang))
    ox, oy = params[3*n], params[3*n+1]
    outer_angle = math.degrees(params[3*n+2]) % 360
    
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

def objective_semirigid(params, use_smooth=True, beta=100.0):
    """Semi-rigid: cx, cy, theta_inner, scale, then individual angle offsets."""
    # params: cx, cy, theta_inner, scale, then N angle offsets
    cx, cy = params[0], params[1]
    theta_inner = params[2]
    scale = params[3]
    
    axial_coords = [
        (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,-1), (1,1), (-2,1), (-1,-1)
    ][:N]
    
    inner = []
    for i, (q, r) in enumerate(axial_coords):
        bx, by = axial_to_cart(q, r, scale)
        bxr = cx + bx * math.cos(theta_inner) - by * math.sin(theta_inner)
        byr = cy + bx * math.sin(theta_inner) + by * math.cos(theta_inner)
        ang_offset = params[4 + i]
        inner.append((bxr, byr, math.degrees(theta_inner + ang_offset) % 360))
    
    # Outer: center at (0,0), angle = theta_inner (for simplicity)
    oa_deg = math.degrees(theta_inner) % 360
    
    if use_smooth:
        S = compute_required_side_smooth(inner, (0.0, 0.0), oa_deg, beta)
    else:
        S = compute_required_side(inner, (0.0, 0.0), oa_deg)
    
    penalty = 0.0
    min_d = SQRT3 * scale
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            d = math.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
            if d < min_d:
                penalty += (min_d - d)**2 * 50.0
    return S + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    best_result = None
    best_score = float('inf')
    
    def try_update(inner, outer_center, outer_angle_deg):
        nonlocal best_result, best_score
        S = compute_required_side(inner, outer_center, outer_angle_deg)
        if check_validity(inner, outer_center, S, outer_angle_deg):
            if S < best_score:
                best_score = S
                best_result = (inner, outer_center, S, outer_angle_deg)
    
    # Phase 1: Rigid composite optimization (6 DOF)
    # Multiple initial conditions for rigid phase
    rigid_ics = []
    
    # IC1: Standard, outer aligned
    rigid_ics.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    # IC2: Outer rotated 30 deg
    rigid_ics.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, math.radians(30)]))
    # IC3: Outer rotated 15 deg
    rigid_ics.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, math.radians(15)]))
    # IC4: Inner rotated 30 deg
    rigid_ics.append(np.array([0.0, 0.0, math.radians(30), 0.0, 0.0, 0.0]))
    # IC5: Inner rotated 30, outer rotated 30
    rigid_ics.append(np.array([0.0, 0.0, math.radians(30), 0.0, 0.0, math.radians(30)]))
    # IC6: Slightly perturbed
    np.random.seed(42)
    rigid_ics.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) + np.random.randn(6)*0.03)
    # IC7: Outer angle 60
    rigid_ics.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, math.radians(60)]))
    # IC8: Inner angle 15
    rigid_ics.append(np.array([0.0, 0.0, math.radians(15), 0.0, 0.0, 0.0]))
    
    for ic in rigid_ics:
        if time.time() > deadline - 30:
            break
        try:
            # Smooth phase
            r1 = minimize(lambda p: objective_rigid(p, True, 50.0), ic, method='L-BFGS-B',
                         options={'maxiter':100, 'ftol':1e-10, 'gtol':1e-8})
            # Higher beta
            r2 = minimize(lambda p: objective_rigid(p, True, 200.0), r1.x, method='L-BFGS-B',
                         options={'maxiter':80, 'ftol':1e-12, 'gtol':1e-10})
            # Exact
            r3 = minimize(lambda p: objective_rigid(p, False), r2.x, method='L-BFGS-B',
                         options={'maxiter':80, 'ftol':1e-14, 'gtol':1e-12})
            
            best_r = min([r1, r2, r3], key=lambda r: r.fun)
            p = best_r.x
            inner = build_rigid_inner(p[0], p[1], p[2], 1.0)
            ox, oy = p[3], p[4]
            oa = math.degrees(p[5]) % 360
            try_update(inner, (ox, oy), oa)
        except Exception:
            pass
    
    # Phase 2: Semi-rigid (allow individual angle offsets)
    if best_result is not None and time.time() < deadline - 25:
        try:
            # Extract best rigid configuration
            inner_b = best_result[0]
            # Estimate rigid params from best
            # Use the first hexagon position to estimate cx, cy
            # For now, just use the best full configuration as starting point
            # Build semi-rigid IC from best result
            sb = np.zeros(4 + N)
            # Estimate: center is roughly the centroid of inner hexes
            cx_est = sum(v[0] for v in inner_b) / N
            cy_est = sum(v[1] for v in inner_b) / N
            sb[0], sb[1] = cx_est, cy_est
            sb[2] = math.radians(inner_b[0][2])  # base angle
            sb[3] = 1.0  # scale
            for i in range(N):
                sb[4+i] = 0.0  # angle offsets
            
            sr1 = minimize(lambda p: objective_semirigid(p, True, 100.0), sb, method='L-BFGS-B',
                          options={'maxiter':150, 'ftol':1e-12, 'gtol':1e-10})
            sr2 = minimize(lambda p: objective_semirigid(p, False), sr1.x, method='L-BFGS-B',
                          options={'maxiter':100, 'ftol':1e-14, 'gtol':1e-12})
            
            best_sr = min([sr1, sr2], key=lambda r: r.fun)
            p = best_sr.x
            # Rebuild inner from semi-rigid params
            axial_coords = [
                (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                (2,-1), (1,1), (-2,1), (-1,-1)
            ][:N]
            cx, cy = p[0], p[1]
            theta_inner = p[2]
            scale = p[3]
            inner_sr = []
            for i, (q, r) in enumerate(axial_coords):
                bx, by = axial_to_cart(q, r, scale)
                bxr = cx + bx * math.cos(theta_inner) - by * math.sin(theta_inner)
                byr = cy + bx * math.sin(theta_inner) + by * math.cos(theta_inner)
                ang_offset = p[4 + i]
                inner_sr.append((bxr, byr, math.degrees(theta_inner + ang_offset) % 360))
            
            oa = math.degrees(theta_inner) % 360
            try_update(inner_sr, (0.0, 0.0), oa)
        except Exception:
            pass
    
    # Phase 3: Full free DOF refinement from best
    if best_result is not None and time.time() < deadline - 15:
        try:
            inner_b = best_result[0]
            oc_b = best_result[1]
            oa_b = best_result[3]
            
            pf = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(inner_b):
                pf[3*i], pf[3*i+1], pf[3*i+2] = cx, cy, math.radians(ang)
            pf[3*N], pf[3*N+1], pf[3*N+2] = oc_b[0], oc_b[1], math.radians(oa_b)
            
            # Smooth
            rf1 = minimize(lambda p: objective_full(p, True, 200.0), pf, method='L-BFGS-B',
                          options={'maxiter':100, 'ftol':1e-12, 'gtol':1e-10})
            # Exact
            rf2 = minimize(lambda p: objective_full(p, False), rf1.x, method='L-BFGS-B',
                          options={'maxiter':100, 'ftol':1e-14, 'gtol':1e-12})
            # Nelder-Mead polish
            rf3 = minimize(lambda p: objective_full(p, False), rf2.x, method='Nelder-Mead',
                          options={'maxiter':300, 'xatol':1e-10, 'fatol':1e-12})
            
            best_rf = min([rf1, rf2, rf3], key=lambda r: r.fun)
            p = best_rf.x
            inner_f = [(p[3*i], p[3*i+1], math.degrees(p[3*i+2])%360) for i in range(N)]
            ox, oy = p[3*N], p[3*N+1]
            oa = math.degrees(p[3*N+2])%360
            try_update(inner_f, (ox, oy), oa)
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if best_result is not None and time.time() < deadline - 5:
        try:
            inner_b = best_result[0]
            oc_b = best_result[1]
            oa_b = best_result[3]
            
            pf = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(inner_b):
                pf[3*i], pf[3*i+1], pf[3*i+2] = cx, cy, math.radians(ang)
            pf[3*N], pf[3*N+1], pf[3*N+2] = oc_b[0], oc_b[1], math.radians(oa_b)
            
            rp = minimize(lambda p: objective_full(p, False), pf, method='L-BFGS-B',
                         options={'maxiter':200, 'ftol':1e-16, 'gtol':1e-14})
            p = rp.x
            inner_p = [(p[3*i], p[3*i+1], math.degrees(p[3*i+2])%360) for i in range(N)]
            ox, oy = p[3*N], p[3*N+1]
            oa = math.degrees(p[3*N+2])%360
            try_update(inner_p, (ox, oy), oa)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        axial_coords = [
            (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,-1), (1,1), (-2,1), (-1,-1)
        ][:N]
        cf = [axial_to_cart(q, r, 1.0) for q, r in axial_coords]
        inner = [(px, py, 0.0) for px, py in cf]
        S = compute_required_side(inner, (0,0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    return best_result

# EVOLVE-BLOCK-END
