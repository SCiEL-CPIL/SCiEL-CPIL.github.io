import numpy as np
import time
from scipy.optimize import minimize

N = 11

_START = time.time()
_DEADLINE = 115.0  # seconds; keep safety margin under 120s

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _hex_vertices(cx, cy, s, ang):
    # regular hexagon, side length s, center (cx,cy), rotation ang (radians)
    a = ang + np.arange(6) * (np.pi / 3.0)
    return np.column_stack([cx + s * np.cos(a), cy + s * np.sin(a)])

def _sat_overlap(A, B):
    # A,B: (6,2) arrays of convex polygon vertices (CCW or CW).
    # Returns penetration depth >0 if overlapping, else <=0 (separated).
    polys = [A, B]
    min_pen = np.inf
    for P in polys:
        for i in range(6):
            p0 = P[i]
            p1 = P[(i + 1) % 6]
            edge = p1 - p0
            # normal
            n = np.array([-edge[1], edge[0]])
            nl = np.linalg.norm(n)
            if nl < 1e-12:
                continue
            n = n / nl
            # project both polys
            a_min, a_max = np.min(P @ n), np.max(P @ n)
            b_min, b_max = np.min(B @ n), np.max(B @ n)
            # overlap on this axis
            overlap = min(a_max, b_max) - max(a_min, b_min)
            if overlap <= 0:
                return 0.0  # separating axis found
            min_pen = min(min_pen, overlap)
    return min_pen

def _all_inside(verts, outer_verts):
    # verts: (6,2); outer_verts: (6,2). Check all 6 vertices inside outer (convex).
    # Fix: Use manual 2D cross product instead of np.cross
    center = np.mean(outer_verts, axis=0)
    for i in range(6):
        p0 = outer_verts[i]
        p1 = outer_verts[(i + 1) % 6]
        edge = p1 - p0
        # ref cross product sign for center relative to edge p0->p1
        ref = edge[0] * (center[1] - p0[1]) - edge[1] * (center[0] - p0[0])
        if abs(ref) < 1e-12:
            continue
        for v in verts:
            c = edge[0] * (v[1] - p0[1]) - edge[1] * (v[0] - p0[0])
            if np.sign(c) != np.sign(ref) and abs(c) > 1e-9:
                return False
    return True

def _objective(x):
    # x: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, S]
    # S is the outer side length. We want to minimize S + penalties.
    positions = x[:N*3].reshape(N, 3)
    S = x[N*3]
    
    # Penalties for overlap and being outside
    penalty = 0.0
    verts_list = []
    for i in range(N):
        cx, cy, ang = positions[i]
        v = _hex_vertices(cx, cy, 1.0, ang)
        verts_list.append(v)
    
    # Outer hexagon centered at 0,0 for simplicity in objective, 
    # but we can also optimize center. Let's fix center at (0,0) and angle 0 
    # for the objective to simplify, and then adjust at the end.
    outer_verts = _hex_vertices(0, 0, S, 0.0)
    
    # Check inside
    for i in range(N):
        if not _all_inside(verts_list[i], outer_verts):
            penalty += 10.0
    
    # Check overlap
    for i in range(N):
        for j in range(i+1, N):
            pen = _sat_overlap(verts_list[i], verts_list[j])
            if pen > 0:
                penalty += 5.0 * pen
                
    return S + penalty

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    
    # Initial layout: hexagonal-ish cluster
    def init_layout():
        pts = [(0.0, 0.0, 0.0)]
        for k in range(6):
            a = k * np.pi / 3.0
            pts.append((2.05 * np.cos(a), 2.05 * np.sin(a), 0.0))
        for k in range(5):
            a = (k + 0.5) * np.pi / 3.0
            pts.append((4.1 * np.cos(a), 4.1 * np.sin(a), 0.0))
        pts = pts[:N]
        return np.array(pts, dtype=float)

    best_x = None
    best_val = np.inf
    
    # Try a few random starts
    for start in range(5):
        if _remaining() < 5.0:
            break
            
        positions = init_layout()
        # Add some noise
        positions[:, 0] += rng.standard_normal(N) * 0.5
        positions[:, 1] += rng.standard_normal(N) * 0.5
        positions[:, 2] += rng.standard_normal(N) * 0.2
        
        S_init = 6.0
        x0 = np.concatenate([positions.flatten(), [S_init]])
        
        # Bounds: positions can be anywhere, S must be positive
        bounds = [(-10, 10), (-10, 10), (-np.pi, np.pi)] * N + [(1.0, 20.0)]
        
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-6, 'gtol': 1e-5})
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
        except Exception:
            pass

    if best_x is None:
        # Fallback: return a trivial valid-ish solution if optimization failed completely
        positions = init_layout()
        S = 10.0
        inner = [[float(positions[i,0]), float(positions[i,1]), float(np.degrees(positions[i,2]))] for i in range(N)]
        return inner, (0.0, 0.0), float(S), 0.0

    # Extract best solution
    positions = best_x[:N*3].reshape(N, 3)
    S = best_x[N*3]
    
    # The optimization assumed outer hex at (0,0) with angle 0.
    # We need to return the inner hexagons relative to some center, and the outer hex center/angle.
    # Since we fixed outer to (0,0), the inner coordinates are already relative to the outer center.
    # So outer center is (0,0) and angle is 0.
    
    # Refine S: binary search for minimal S that fits the found configuration
    def fits_configuration(S_test, oc, oa):
        ov = _hex_vertices(oc[0], oc[1], S_test, oa)
        for i in range(N):
            v = _hex_vertices(positions[i,0], positions[i,1], 1.0, positions[i,2])
            if not _all_inside(v, ov):
                return False
        # Check overlaps are zero (they should be from optimization, but verify)
        verts_list = []
        for i in range(N):
            v = _hex_vertices(positions[i,0], positions[i,1], 1.0, positions[i,2])
            verts_list.append(v)
        for i in range(N):
            for j in range(i+1, N):
                if _sat_overlap(verts_list[i], verts_list[j]) > 1e-6:
                    return False
        return True

    best_side = S
    # Try to shrink S
    lo, hi = 1.0, S
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if fits_configuration(mid, np.array([0.0, 0.0]), 0.0):
            hi = mid
        else:
            lo = mid
    
    if hi < best_side:
        best_side = hi

    inner = [[float(positions[i,0]), float(positions[i,1]), float(np.degrees(positions[i,2]))] for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), 0.0
