"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
INRADIUS = np.sqrt(3) / 2.0
MIN_DIST = np.sqrt(3) - 1e-5 # Minimum center distance for non-overlapping unit hexagons

def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _is_valid(inner_pts, outer_center, side, angle_deg):
    """Check if inner hexagons are disjoint and contained in outer hexagon."""
    # Check overlaps
    verts = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner_pts]
    for i in range(N):
        for j in range(i+1, N):
            # SAT check
            v1, v2 = verts[i], verts[j]
            for v in [v1, v2]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k+1)%6]
                    edge = p2 - p1
                    nx, ny = -edge[1], edge[0]
                    norm = np.sqrt(nx*nx + ny*ny)
                    if norm < 1e-9: continue
                    nx, ny = nx/norm, ny/norm
                    proj1 = v1 @ np.array([nx, ny])
                    proj2 = v2 @ np.array([nx, ny])
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 < min2 or max2 < min1:
                        continue # Separated in this axis
                    # Overlap in this axis, check next
                    break
                else:
                    # No separating axis found for this pair
                    return False
    # Check containment
    theta = np.radians(angle_deg)
    face_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + theta
    normals = np.column_stack([np.cos(face_angles), np.sin(face_angles)])
    apothem = side * INRADIUS
    
    for p in inner_pts:
        v = _hex_vertices(p[0], p[1], 1.0, p[2])
        # Translate vertices to be relative to outer center
        v_rel = v - np.array(outer_center)
        for n in normals:
            proj = np.max(v_rel @ n)
            if proj > apothem + 1e-6:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Initial valid configuration: 3x4 grid centered at origin
    # This ensures we start with a valid solution
    def make_initial():
        pts = []
        dx = 2 * np.sqrt(3)
        dy = 1.5 * np.sqrt(3)
        for r in range(3):
            for c in range(4):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += dx / 2
                pts.append([x, y, 0.0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in pts:
            p[0] -= cx
            p[1] -= cy
        return pts

    initial_pts = make_initial()
    # Calculate initial side length
    # We need to find min side for this config.
    # We'll just use a large enough side initially and let optimizer shrink it.
    # Or compute it.
    def compute_min_side(pts, angle_deg=0.0):
        theta = np.radians(angle_deg)
        face_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + theta
        normals = np.column_stack([np.cos(face_angles), np.sin(face_angles)])
        min_s = 0.0
        for p in pts:
            v = _hex_vertices(p[0], p[1], 1.0, p[2])
            for n in normals:
                max_proj = np.max(v @ n)
                s_needed = max_proj / INRADIUS
                if s_needed > min_s:
                    min_s = s_needed
        return min_s

    init_side = compute_min_side(initial_pts)
    best_side = init_side
    best_inner = [p[:] for p in initial_pts]
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Optimization variables:
    # x[0:36] : x, y, angle for 12 hexes
    # x[36:38] : outer center cx, cy
    # x[38]    : outer angle (radians)
    # x[39]    : outer side length (minimize this)
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        ix = x[0:36].reshape(N, 3)
        cx, cy = x[36], x[37]
        angle_rad = x[38]
        side = x[39]
        
        # Objective is side length
        # We enforce constraints via SLSQP
        return side

    def constraints(x):
        if time.time() > deadline:
            return [{} for _ in range(100)] # Dummy
        ix = x[0:36].reshape(N, 3)
        cx, cy = x[36], x[37]
        angle_rad = x[38]
        side = x[39]
        
        cons = []
        
        # 1. Non-overlap constraints: dist >= MIN_DIST
        # dist^2 - MIN_DIST^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                dx = ix[i,0] - ix[j,0]
                dy = ix[i,1] - ix[j,1]
                dist_sq = dx*dx + dy*dy
                cons.append({'type': 'ineq', 'fun': lambda dist_sq=dist_sq: dist_sq - MIN_DIST**2})
        
        # 2. Containment constraints:
        # For each inner hex i and each outer normal k:
        # max_proj(v_i, n_k) - side * INRADIUS <= 0
        # This is equivalent to: side * INRADIUS - max_proj(v_i, n_k) >= 0
        theta = angle_rad
        face_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + theta
        normals = np.column_stack([np.cos(face_angles), np.sin(face_angles)])
        
        for i in range(N):
            ix_i, iy_i, ia_i = ix[i]
            v = _hex_vertices(ix_i, iy_i, 1.0, np.degrees(ia_i))
            v_rel = v - np.array([cx, cy])
            for k in range(6):
                n = normals[k]
                max_proj = np.max(v_rel @ n)
                # Constraint: side * INRADIUS - max_proj >= 0
                cons.append({'type': 'ineq', 'fun': lambda side=side, max_proj=max_proj: side * INRADIUS - max_proj})
        
        return cons

    # Run SLSQP
    # Initial guess
    x0 = []
    for p in initial_pts:
        x0.extend([p[0], p[1], np.radians(p[2])])
    x0.extend([0.0, 0.0, 0.0, init_side])
    
    try:
        res = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                       options={'maxiter': 200, 'ftol': 1e-6, 'disp': False})
        
        if res.success and res.fun < best_side:
            # Extract and validate strictly
            ix = res.x[0:36].reshape(N, 3)
            cx, cy = res.x[36], res.x[37]
            angle_deg = np.degrees(res.x[38])
            side = res.x[39]
            
            inner_pts = [[ix[i,0], ix[i,1], np.degrees(ix[i,2])] for i in range(N)]
            
            if _is_valid(inner_pts, (cx, cy), side, angle_deg):
                best_side = side
                best_inner = inner_pts
                best_center = (cx, cy)
                best_angle = angle_deg
    except Exception as e:
        pass

    # If SLSQP failed or didn't improve, we still have the initial valid solution
    # We can try a few random restarts with SLSQP if time permits
    if time.time() < deadline - 20.0:
        for restart in range(3):
            # Perturb initial config
            pts = [p[:] for p in initial_pts]
            for p in pts:
                p[0] += np.random.uniform(-0.5, 0.5)
                p[1] += np.random.uniform(-0.5, 0.5)
                p[2] += np.random.uniform(-10, 10)
            
            # Check if this perturbed config is valid with a larger side
            # We don't need to check validity yet, just use it as start
            # But we need a valid side for start. Let's use a large side.
            start_side = 20.0
            
            x0_r = []
            for p in pts:
                x0_r.extend([p[0], p[1], np.radians(p[2])])
            x0_r.extend([0.0, 0.0, 0.0, start_side])
            
            try:
                res_r = minimize(objective, x0_r, method='SLSQP', constraints=constraints, 
                                 options={'maxiter': 100, 'ftol': 1e-5, 'disp': False})
                
                if res_r.success and res_r.fun < best_side:
                    ix = res_r.x[0:36].reshape(N, 3)
                    cx, cy = res_r.x[36], res_r.x[37]
                    angle_deg = np.degrees(res_r.x[38])
                    side = res_r.x[39]
                    
                    inner_pts = [[ix[i,0], ix[i,1], np.degrees(ix[i,2])] for i in range(N)]
                    
                    if _is_valid(inner_pts, (cx, cy), side, angle_deg):
                        best_side = side
                        best_inner = inner_pts
                        best_center = (cx, cy)
                        best_angle = angle_deg
            except Exception:
                pass

    return best_inner, best_center, best_side, best_angle

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side}")
