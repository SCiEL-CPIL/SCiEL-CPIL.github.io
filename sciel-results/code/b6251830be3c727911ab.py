"""Combined program: radial-ring lattice packing with LP enclosing-hex and SLSQP polish."""
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
    """For a fixed outer hexagon angle, find optimal center and side length using LP.
    Returns (cx, cy, side_length)."""
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0.0, 0.0, 0.0
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            # (v - c) . n <= R  =>  -n.x*cx - n.y*cy - R <= -n.x*vx - n.y*vy
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
            # -(v - c) . n <= R =>  n.x*cx + n.y*cy - R <=  n.x*vx + n.y*vy
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)
    
    # R >= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, 1.0])
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                     bounds=[(None, None), (None, None), (0, None)], method='highs')
    
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
    return cx_opt, cy_opt, side

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
    
    # Coarse sweep over outer angle (0 to 60)
    for angle_deg in np.arange(0, 60, 1.0):
        if time.time() > _DEADLINE - 10:
            break
        cx, cy, side = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Fine sweep
    if best_side < float('inf') and time.time() < _DEADLINE - 15:
        for angle_deg in np.arange(best_angle - 3, best_angle + 3, 0.1):
            if angle_deg < 0 or angle_deg >= 60:
                continue
            cx, cy, side = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Even finer
    if best_side < float('inf') and time.time() < _DEADLINE - 15:
        for angle_deg in np.arange(best_angle - 0.5, best_angle + 0.5, 0.01):
            if angle_deg < 0 or angle_deg >= 60:
                continue
            cx, cy, side = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
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
            if res.fun < best_side:
                cx_opt, cy_opt, ang_opt = res.x
                ang_opt = ang_opt % 60.0
                best_side = res.fun
                best_center = (cx_opt, cy_opt)
                best_angle = ang_opt
        except Exception:
            pass
        
        # Ultra-fine sweep
        if best_side < float('inf') and time.time() < _DEADLINE - 10:
            for angle_deg in np.arange(best_angle - 0.1, best_angle + 0.1, 0.001):
                if angle_deg < 0 or angle_deg >= 60:
                    continue
                cx, cy, side = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
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
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx*dx + dy*dy)
            # Quick rejection: if center distance < sqrt(3) - gap, definitely overlap
            if dist < math.sqrt(3) - gap:
                return False
            # If close, do a more careful check
            if dist < 2.0 + gap:
                v1 = _get_hex_vertices(x1, y1, a1)
                v2 = _get_hex_vertices(x2, y2, a2)
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

def _generate_radial_ring_packing(ring_scale=1.001, inner_angle=30.0):
    """Generate 11 hexagons using a radial-ring structure.
    
    Structure: 1 center + 6 inner ring + 4 outer ring (partial).
    The inner ring is at distance sqrt(3)*scale, the outer ring at 2*sqrt(3)*scale.
    We optimize the radial distances and angular offsets.
    """
    sqrt3 = math.sqrt(3)
    
    # Ring 0: center
    # Ring 1: 6 hexagons at radius r1, angles 0, 60, 120, 180, 240, 300
    # Ring 2: 4 hexagons at radius r2, angles 30, 150, 210, 330 (between ring-1 positions)
    
    r1 = sqrt3 * ring_scale
    r2 = 2 * sqrt3 * ring_scale
    
    positions = []
    positions.append((0.0, 0.0))  # center
    
    for k in range(6):
        a = math.radians(60 * k)
        x = r1 * math.cos(a)
        y = r1 * math.sin(a)
        positions.append((x, y))
    
    # 4 outer ring hexagons at angles 30, 150, 210, 330
    for k in range(4):
        a = math.radians(30 + 120 * k)
        x = r2 * math.cos(a)
        y = r2 * math.sin(a)
        positions.append((x, y))
    
    return [[pos[0], pos[1], inner_angle] for pos in positions]

def _optimize_radial_packing(max_iters=100):
    """Optimize the radial ring packing parameters: r1, r2, angular offsets."""
    sqrt3 = math.sqrt(3)
    
    def objective(params):
        if time.time() > _DEADLINE - 5:
            return 1e10
        r1, r2, ang_offset1, ang_offset2 = params
        r1 = max(r1, sqrt3 * 0.99)
        r2 = max(r2, 2 * sqrt3 * 0.99)
        
        positions = []
        positions.append((0.0, 0.0))
        
        for k in range(6):
            a = math.radians(60 * k + ang_offset1)
            x = r1 * math.cos(a)
            y = r1 * math.sin(a)
            positions.append((x, y))
        
        for k in range(4):
            a = math.radians(30 + 120 * k + ang_offset2)
            x = r2 * math.cos(a)
            y = r2 * math.sin(a)
            positions.append((x, y))
        
        # Check disjointness
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                d = math.sqrt(dx*dx + dy*dy)
                if d < math.sqrt(3) - 0.01:
                    return 1e10
        
        # Compute enclosing hex side (use fixed outer angle for speed in this inner loop)
        all_verts = []
        for x, y in positions:
            verts = _get_hex_vertices(x, y, 30.0)
            all_verts.extend(verts)
        
        # Try a few outer angles
        best_r = 1e10
        for outer_angle in [0.0, 15.0, 30.0, 45.0]:
            r_max = 0.0
            for k in range(6):
                a = math.radians(outer_angle + 30 + 60 * k)
                nx = math.cos(a)
                ny = math.sin(a)
                cx = sum(v[0] for v in all_verts) / len(all_verts)
                cy = sum(v[1] for v in all_verts) / len(all_verts)
                for vx, vy in all_verts:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    if proj > r_max:
                        r_max = proj
            if r_max < best_r:
                best_r = r_max
        
        return best_r / (math.sqrt(3) / 2)
    
    x0 = np.array([sqrt3 * 1.001, 2 * sqrt3 * 1.001, 0.0, 0.0])
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < 1e9:
            r1, r2, a1, a2 = res.x
            positions = []
            positions.append((0.0, 0.0))
            for k in range(6):
                a = math.radians(60 * k + a1)
                x = r1 * math.cos(a)
                y = r1 * math.sin(a)
                positions.append((x, y))
            for k in range(4):
                a = math.radians(30 + 120 * k + a2)
                x = r2 * math.cos(a)
                y = r2 * math.sin(a)
                positions.append((x, y))
            inner = [[pos[0], pos[1], 30.0] for pos in positions]
            if _hexagons_disjoint(inner, gap=0.002):
                return inner
    except Exception:
        pass
    
    return None

def _optimize_positions(inner_hexes, outer_angle, max_iters=200):
    """Try to optimize inner hexagon positions to minimize outer hex side length."""
    cx0, cy0, side0 = _min_enclosing_hex_side_lp(inner_hexes, outer_angle)
    
    n_params = 3 * N
    
    def objective(params):
        if time.time() > _DEADLINE - 5:
            return 1e10
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
            if _hexagons_disjoint(positions, gap=0.002):
                return positions, res.fun
    except Exception:
        pass
    
    return inner_hexes, side0

def _slsqp_polish(inner_hexes, outer_angle, max_iters=300):
    """High-precision SLSQP polish pass for the inner hexagon positions."""
    cx0, cy0, side0 = _min_enclosing_hex_side_lp(inner_hexes, outer_angle)
    n_params = 3 * N
    
    def objective(params):
        if time.time() > _DEADLINE - 3:
            return 1e10
        positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2] % 360.0
            positions.append((x, y, a))
        
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
    
    # Constraints: pairwise distance >= sqrt(3) - small gap
    cons = []
    for i in range(N):
        for j in range(i+1, N):
            def make_cons(i, j):
                def constraint(params):
                    xi = params[3*i]
                    yi = params[3*i+1]
                    xj = params[3*j]
                    yj = params[3*j+1]
                    d = math.sqrt((xj-xi)**2 + (yj-yi)**2)
                    return d - (math.sqrt(3) - 0.005)
                return constraint
            cons.append({'type': 'ineq', 'fun': make_cons(i, j)})
    
    x0 = np.zeros(n_params)
    for i in range(N):
        x0[3*i] = inner_hexes[i][0]
        x0[3*i+1] = inner_hexes[i][1]
        x0[3*i+2] = inner_hexes[i][2]
    
    try:
        res = minimize(objective, x0, method='SLSQP',
                      bounds=[(None, None)] * n_params,
                      constraints=cons,
                      options={'maxiter': max_iters, 'ftol': 1e-12})
        if res.success and res.fun < side0:
            positions = []
            for i in range(N):
                positions.append([res.x[3*i], res.x[3*i+1], res.x[3*i+2] % 360.0])
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
    
    # Strategy 1: Standard hexagonal lattice cluster with multiple scales/angles
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
        if time.time() > _DEADLINE - 20:
            break
        
        inner = _generate_hexagonal_cluster(scale, angle)
        
        if not _hexagons_disjoint(inner, gap=0.0005):
            continue
        
        cx, cy, side, ang = _min_enclosing_hex_side(inner)
        
        if side < best_side:
            best_side = side
            best_inner = inner
            best_center = (cx, cy)
            best_angle = ang
    
    # Strategy 2: Radial ring packing
    if time.time() < _DEADLINE - 25:
        radial_inner = _generate_radial_packing(1.001, 30.0)
        if radial_inner and _hexagons_disjoint(radial_inner, gap=0.0005):
            cx, cy, side, ang = _min_enclosing_hex_side(radial_inner)
            if side < best_side:
                best_side = side
                best_inner = radial_inner
                best_center = (cx, cy)
                best_angle = ang
    
    # Strategy 3: Optimize radial packing parameters
    if time.time() < _DEADLINE - 30:
        opt_radial = _optimize_radial_packing(max_iters=80)
        if opt_radial and _hexagons_disjoint(opt_radial, gap=0.0005):
            cx, cy, side, ang = _min_enclosing_hex_side(opt_radial)
            if side < best_side:
                best_side = side
                best_inner = opt_radial
                best_center = (cx, cy)
                best_angle = ang
    
    # Strategy 4: L-BFGS-B position optimization on best config
    if best_inner is not None and time.time() < _DEADLINE - 30:
        opt_inner, opt_side = _optimize_positions(best_inner, best_angle, max_iters=200)
        if opt_side < best_side:
            cx, cy, side, ang = _min_enclosing_hex_side(opt_inner)
            if side < best_side:
                best_side = side
                best_inner = opt_inner
                best_center = (cx, cy)
                best_angle = ang
    
    # Strategy 5: SLSQP high-precision polish
    if best_inner is not None and time.time() < _DEADLINE - 15:
        polished_inner, polished_side = _slsqp_polish(best_inner, best_angle, max_iters=200)
        if polished_side < best_side:
            cx, cy, side, ang = _min_enclosing_hex_side(polished_inner)
            if side < best_side:
                best_side = side
                best_inner = polished_inner
                best_center = (cx, cy)
                best_angle = ang
    
    if best_inner is None:
        best_inner = _generate_hexagonal_cluster(1.001, 30.0)
        best_side = 40.0
        best_center = (0.0, 0.0)
        best_angle = 0.0
    
    return best_inner, best_center, best_side, best_angle
