"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 115s safety margin within 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """For a fixed outer hexagon angle, find optimal center and side length using LP."""
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-vx * nx - vy * ny)
    
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, 1.0])
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if result.success:
        cx_opt = result.x[0]
        cy_opt = result.x[1]
        r_opt = result.x[2]
    else:
        cx_mean = sum(v[0] for v in all_vertices) / n_pts
        cy_mean = sum(v[1] for v in all_vertices) / n_pts
        r_opt = 0.0
        for k in range(6):
            a = math.radians(angle_deg + 30 + 60 * k)
            nx = math.cos(a)
            ny = math.sin(a)
            for vx, vy in all_vertices:
                proj = (vx - cx_mean) * nx + (vy - cy_mean) * ny
                r_opt = max(r_opt, proj)
        cx_opt = cx_mean
        cy_opt = cy_mean
    
    side = r_opt / (math.sqrt(3) / 2)
    return cx_opt, cy_opt, side, angle_deg

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Coarse sweep
    for angle_deg in np.arange(0, 60, 1.0):
        if time.time() > _DEADLINE - 10:
            break
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Fine sweep
    if best_side < float('inf') and time.time() < _DEADLINE - 15:
        for angle_deg in np.arange(best_angle - 3, best_angle + 3, 0.1):
            if angle_deg < 0 or angle_deg >= 60:
                continue
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Even finer
    if best_side < float('inf') and time.time() < _DEADLINE - 15:
        for angle_deg in np.arange(best_angle - 0.5, best_angle + 0.5, 0.01):
            if angle_deg < 0 or angle_deg >= 60:
                continue
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Nelder-Mead polish
    if best_side < float('inf') and time.time() < _DEADLINE - 20:
        def objective(params):
            cx, cy, angle_deg = params
            angle_deg = angle_deg % 60.0
            r_max = 0.0
            for k in range(6):
                a = math.radians(angle_deg + 30 + 60 * k)
                nx = math.cos(a)
                ny = math.sin(a)
                for vx, vy in all_vertices:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    if proj > r_max:
                        r_max = proj
            return r_max / (math.sqrt(3) / 2)
        
        x0 = np.array([best_center[0], best_center[1], best_angle])
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.success:
                cx_opt, cy_opt, ang_opt = res.x
                ang_opt = ang_opt % 60.0
                side_opt = res.fun
                if side_opt < best_side:
                    best_side = side_opt
                    best_center = (cx_opt, cy_opt)
                    best_angle = ang_opt
        except Exception:
            pass
        
        # Ultra-fine sweep
        if best_side < float('inf') and time.time() < _DEADLINE - 10:
            for angle_deg in np.arange(best_angle - 0.1, best_angle + 0.1, 0.001):
                if angle_deg < 0 or angle_deg >= 60:
                    continue
                cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)
    
    return best_center[0], best_center[1], best_side, best_angle

def _hexagons_disjoint(inner_hexes, gap=0.001):
    """Check if all pairs of hexagons are disjoint (with a small gap)."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = inner_hexes[i]
            x2, y2, a2 = inner_hexes[j]
            # Center distance
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx*dx + dy*dy)
            # Minimum distance between two unit hexagons depends on relative angle
            # For same orientation, minimum center distance is sqrt(3) (flat-to-flat)
            # For 30 deg relative, minimum is 2 (vertex-to-vertex)
            # Conservative: require dist > 2 - gap (covers all cases)
            # Actually, the minimum possible distance between centers of two disjoint
            # unit hexagons is sqrt(3) ≈ 1.732 (when aligned flat-to-flat)
            # Maximum minimum distance is 2 (vertex-to-vertex)
            # So we need dist >= min_dist(a1, a2) + gap
            # For simplicity, use the worst case: sqrt(3) is the absolute minimum
            # But to be safe, check actual vertex-vertex distances
            if dist < 2.0 - gap:
                # Could still be disjoint if oriented favorably
                # Quick check: if dist < sqrt(3) + gap, definitely overlap for any orientation
                if dist < math.sqrt(3) + gap:
                    return False
                # Otherwise do a more careful check
                v1 = _get_hex_vertices(x1, y1, a1)
                v2 = _get_hex_vertices(x2, y2, a2)
                # Check if any vertex of one is inside the other
                for vx, vy in v1:
                    if _point_in_hex(vx, vy, x2, y2, a2):
                        return False
                for vx, vy in v2:
                    if _point_in_hex(vx, vy, x1, y1, a1):
                        return False
    return True

def _point_in_hex(px, py, cx, cy, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given angle."""
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        if (px - cx) * nx + (py - cy) * ny > 1.0 + 1e-9:
            return False
    return True

def _generate_hexagonal_cluster(scale=1.001, angle=30.0):
    """Generate N hexagons in a hexagonal cluster on triangular lattice."""
    sqrt3 = math.sqrt(3)
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
    
    return [[pos[0], pos[1], angle] for pos in positions]

def _optimize_positions(inner_hexes, outer_angle, max_iters=200):
    """Try to optimize inner hexagon positions to minimize outer hex side length."""
    # Use the LP result as a baseline
    cx0, cy0, side0, ang0 = _min_enclosing_hex_side_lp(inner_hexes, outer_angle)
    
    # Parameters: for each hexagon, (x, y, angle)
    # Total params: 3*N
    n_params = 3 * N
    
    def objective(params):
        if time.time() > _DEADLINE - 5:
            return 1e10
        # Decode params
        positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2] % 360.0
            positions.append((x, y, a))
        
        # Check disjointness (quick)
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                d = math.sqrt(dx*dx + dy*dy)
                if d < math.sqrt(3) - 0.05:
                    return 1e10
        
        # Compute enclosing hex side for fixed outer angle
        all_verts = []
        for x, y, a in positions:
            verts = _get_hex_vertices(x, y, a)
            all_verts.extend(verts)
        
        r_max = 0.0
        for k in range(6):
            a_rad = math.radians(outer_angle + 30 + 60 * k)
            nx = math.cos(a_rad)
            ny = math.sin(a_rad)
            for vx, vy in all_verts:
                proj = (vx - cx0) * nx + (vy - cy0) * ny
                if proj > r_max:
                    r_max = proj
        
        if r_max < 0:
            r_max = 0
        return r_max / (math.sqrt(3) / 2)
    
    # Initial params
    x0 = np.zeros(n_params)
    for i in range(N):
        x0[3*i] = inner_hexes[i][0]
        x0[3*i+1] = inner_hexes[i][1]
        x0[3*i+2] = inner_hexes[i][2]
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < side0:
            positions = []
            for i in range(N):
                positions.append([res.x[3*i], res.x[3*i+1], res.x[3*i+2] % 360.0])
            # Verify disjointness
            if _hexagons_disjoint(positions, gap=0.002):
                return positions, res.fun
    except Exception:
        pass
    
    return inner_hexes, side0

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = _generate_hexagonal_cluster(1.001, 30.0)
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_inner = None
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try multiple configurations
    configs = [
        (1.001, 30.0),
        (1.001, 0.0),
        (1.0005, 30.0),
        (1.0001, 30.0),
        (1.0001, 0.0),
        (1.0, 30.0),
        (1.0, 0.0),
    ]
    
    for scale, angle in configs:
        if time.time() > _DEADLINE - 15:
            break
        
        inner = _generate_hexagonal_cluster(scale, angle)
        
        # Check disjointness
        if not _hexagons_disjoint(inner, gap=0.0005):
            continue
        
        cx, cy, side, ang = _min_enclosing_hex_side(inner)
        
        if side < best_side:
            best_side = side
            best_inner = inner
            best_center = (cx, cy)
            best_angle = ang
    
    # Try optimization on the best configuration
    if best_inner is not None and time.time() < _DEADLINE - 30:
        opt_inner, opt_side = _optimize_positions(best_inner, best_angle, max_iters=300)
        if opt_side < best_side:
            # Recompute enclosing hex for optimized positions
            cx, cy, side, ang = _min_enclosing_hex_side(opt_inner)
            if side < best_side:
                best_side = side
                best_inner = opt_inner
                best_center = (cx, cy)
                best_angle = ang
    
    # Try a few more scales around the best
    if best_inner is not None and time.time() < _DEADLINE - 20:
        base_scale = 1.0005
        base_angle = best_inner[0][2]
        for ds in [-0.001, -0.0005, 0.0005, 0.001]:
            if time.time() > _DEADLINE - 10:
                break
            s = base_scale + ds
            if s <= 0.999:
                continue
            inner = _generate_hexagonal_cluster(s, base_angle)
            if not _hexagons_disjoint(inner, gap=0.0005):
                continue
            cx, cy, side, ang = _min_enclosing_hex_side(inner)
            if side < best_side:
                best_side = side
                best_inner = inner
                best_center = (cx, cy)
                best_angle = ang
    
    if best_inner is None:
        best_inner = _generate_hexagonal_cluster(1.001, 30.0)
        best_side = 40.0
        best_center = (0.0, 0.0)
        best_angle = 0.0
    
    return best_inner, best_center, best_side, best_angle
