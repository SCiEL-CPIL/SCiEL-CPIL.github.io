"""Packing 11 unit hexagons in a regular hexagon; minimize outer side length."""
import math
import time
import numpy as np

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin within 150s limit

def _compute_all_vertices(inner_hexes):
    """Precompute all vertices of inner hexagons as a numpy array (N*6, 2)."""
    n_inner = len(inner_hexes)
    verts = np.zeros((n_inner * 6, 2))
    for idx, (ix, iy, iangle) in enumerate(inner_hexes):
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + math.pi * k / 3
            verts[idx * 6 + k, 0] = ix + math.cos(a)
            verts[idx * 6 + k, 1] = iy + math.sin(a)
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg, all_vertices=None):
    """For a fixed outer hexagon angle, find optimal center and side length using LP."""
    if all_vertices is None:
        all_vertices = _compute_all_vertices(inner_hexes)
    
    n_pts = all_vertices.shape[0]
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg
    
    # For each of 6 directions, compute normals
    angles = np.array([math.radians(angle_deg + 30 + 60 * k) for k in range(6)])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
    
    # Vectorized constraint construction
    rhs = -(all_vertices @ normals.T)  # (n_pts, 6)
    b_ub = rhs.T.flatten()  # (6 * n_pts,)
    
    A_ub_parts = np.column_stack([
        np.repeat(-normals[:, 0], n_pts),
        np.repeat(-normals[:, 1], n_pts),
        np.full(6 * n_pts, -1.0)
    ])
    
    # Add r >= 0 constraint
    A_ub = np.vstack([A_ub_parts, [[0.0, 0.0, -1.0]]])
    b_ub = np.append(b_ub, 0.0)
    
    c = np.array([0.0, 0.0, 1.0])  # minimize r
    
    from scipy.optimize import linprog
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if result.success:
        cx_opt = result.x[0]
        cy_opt = result.x[1]
        r_opt = result.x[2]
    else:
        cx_mean = all_vertices[:, 0].mean()
        cy_mean = all_vertices[:, 1].mean()
        centered = all_vertices - np.array([cx_mean, cy_mean])
        projs = centered @ normals.T
        r_opt = projs.max()
        if r_opt < 0:
            r_opt = 0.0
        cx_opt = cx_mean
        cy_opt = cy_mean
    
    side = r_opt / (math.sqrt(3) / 2)
    return cx_opt, cy_opt, side, angle_deg

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    all_vertices = _compute_all_vertices(inner_hexes)
    
    # Coarse sweep
    angles_coarse = np.arange(0, 60, 2.0)
    for angle_deg in angles_coarse:
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Fine sweep
    if best_side < float('inf'):
        angles_fine = np.arange(max(0.0, best_angle - 5), min(60.0, best_angle + 5), 0.25)
        for angle_deg in angles_fine:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Even finer sweep
    if best_side < float('inf') and time.time() < _DEADLINE - 20.0:
        angles_finer = np.arange(max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0), 0.05)
        for angle_deg in angles_finer:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Very fine sweep
    if best_side < float('inf') and time.time() < _DEADLINE - 15.0:
        angles_very_fine = np.arange(max(0.0, best_angle - 0.5), min(60.0, best_angle + 0.5), 0.01)
        for angle_deg in angles_very_fine:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Bounded scalar minimization polish
    if best_side < float('inf') and time.time() < _DEADLINE - 10.0:
        from scipy.optimize import minimize_scalar
        
        def objective_angle(ang_p):
            ang_p = ang_p % 60.0
            _, _, side_p, _ = _min_enclosing_hex_side_lp(inner_hexes, float(ang_p), all_vertices)
            return side_p
        
        try:
            bounds = (max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0))
            res = minimize_scalar(objective_angle, bounds=bounds, method='bounded',
                                  options={'xatol': 1e-10, 'maxiter': 200})
            if res.success:
                ang_opt = float(res.x % 60.0)
                cx_v, cy_v, side_v, _ = _min_enclosing_hex_side_lp(inner_hexes, ang_opt, all_vertices)
                if side_v < best_side:
                    best_side = side_v
                    best_center = (cx_v, cy_v)
                    best_angle = ang_opt
        except Exception:
            pass
    
    # Final micro-adjustments
    if best_side < float('inf') and time.time() < _DEADLINE - 5.0:
        for delta in [0.001, -0.001, 0.002, -0.002, 0.005, -0.005, 0.01, -0.01]:
            ang_test = (best_angle + delta) % 60.0
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(ang_test), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(ang_test)
    
    return best_center[0], best_center[1], best_side, best_angle

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    sqrt3 = math.sqrt(3)
    
    best_result = None
    best_side = float('inf')
    
    # Configuration 1: Standard triangular lattice, pointy-top (30 deg), tight packing
    for scale in [1.001, 1.005, 1.01]:
        u = (sqrt3 * scale, 0.0)
        v = (sqrt3 / 2 * scale, 1.5 * scale)
        
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                dist_sq = x * x + y * y
                candidates.append((dist_sq, x, y))
        
        candidates.sort(key=lambda t: t[0])
        
        positions = []
        for _, x, y in candidates:
            positions.append((x, y))
            if len(positions) == N:
                break
        
        inner = [[pos[0], pos[1], 30.0] for pos in positions]
        
        if time.time() > _DEADLINE - 15.0:
            # Quick estimate
            xs = [h[0] for h in inner]
            ys = [h[1] for h in inner]
            cx = (max(xs) + min(xs)) / 2
            cy = (max(ys) + min(ys)) / 2
            max_dist = 0
            for h in inner:
                d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
                if d > max_dist:
                    max_dist = d
            side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
            if side < best_side:
                best_side = side
                best_result = (inner, (cx, cy), side, 0.0)
            continue
        
        cx, cy, side, angle = _min_enclosing_hex_side(inner)
        
        if side < best_side:
            best_side = side
            best_result = (inner, (cx, cy), side, angle)
    
    # Configuration 2: Try flat-top (0 deg) inner hexagons
    for scale in [1.001, 1.005]:
        u = (sqrt3 * scale, 0.0)
        v = (sqrt3 / 2 * scale, 1.5 * scale)
        
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                dist_sq = x * x + y * y
                candidates.append((dist_sq, x, y))
        
        candidates.sort(key=lambda t: t[0])
        
        positions = []
        for _, x, y in candidates:
            positions.append((x, y))
            if len(positions) == N:
                break
        
        inner = [[pos[0], pos[1], 0.0] for pos in positions]
        
        if time.time() > _DEADLINE - 15.0:
            xs = [h[0] for h in inner]
            ys = [h[1] for h in inner]
            cx = (max(xs) + min(xs)) / 2
            cy = (max(ys) + min(ys)) / 2
            max_dist = 0
            for h in inner:
                d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
                if d > max_dist:
                    max_dist = d
            side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
            if side < best_side:
                best_side = side
                best_result = (inner, (cx, cy), side, 0.0)
            continue
        
        cx, cy, side, angle = _min_enclosing_hex_side(inner)
        
        if side < best_side:
            best_side = side
            best_result = (inner, (cx, cy), side, angle)
    
    # Configuration 3: Try a slightly rotated arrangement
    # Rotate the entire cluster by a small angle
    for rot_deg in [0.0, 5.0, 10.0, 15.0, 30.0]:
        if time.time() > _DEADLINE - 20.0:
            break
        
        rot_rad = math.radians(rot_deg)
        scale = 1.005
        u = (sqrt3 * scale, 0.0)
        v = (sqrt3 / 2 * scale, 1.5 * scale)
        
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                dist_sq = x * x + y * y
                candidates.append((dist_sq, x, y))
        
        candidates.sort(key=lambda t: t[0])
        
        positions = []
        for _, x, y in candidates:
            positions.append((x, y))
            if len(positions) == N:
                break
        
        # Rotate positions and adjust inner hexagon angles
        inner = []
        for (px, py) in positions:
            rx = px * math.cos(rot_rad) - py * math.sin(rot_rad)
            ry = px * math.sin(rot_rad) + py * math.cos(rot_rad)
            inner.append([rx, ry, 30.0 + rot_deg])
        
        if time.time() > _DEADLINE - 15.0:
            xs = [h[0] for h in inner]
            ys = [h[1] for h in inner]
            cx = (max(xs) + min(xs)) / 2
            cy = (max(ys) + min(ys)) / 2
            max_dist = 0
            for h in inner:
                d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
                if d > max_dist:
                    max_dist = d
            side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
            if side < best_side:
                best_side = side
                best_result = (inner, (cx, cy), side, 0.0)
            continue
        
        cx, cy, side, angle = _min_enclosing_hex_side(inner)
        
        if side < best_side:
            best_side = side
            best_result = (inner, (cx, cy), side, angle)
    
    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
