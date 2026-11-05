"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, linprog

N = 12
TIME_LIMIT = 145.0

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB, tol=1e-9):
    """Return True if two convex polygons overlap."""
    # Quick center distance check
    cxA = sum(v[0] for v in vertsA) / 6
    cyA = sum(v[1] for v in vertsA) / 6
    cxB = sum(v[0] for v in vertsB) / 6
    cyB = sum(v[1] for v in vertsB) / 6
    if math.hypot(cxA - cxB, cyA - cyB) > 4.0:
        return False
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + tol or maxB <= minA + tol:
            return False
    return True

def _min_overlap_depth(vertsA, vertsB):
    """Compute minimum overlap depth (0 if disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    max_overlap = 0.0
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap > max_overlap:
            max_overlap = overlap
        if overlap < -1e-9:
            return 0.0
    return max_overlap

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles, tol=1e-9):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j], tol):
                return False
    return True

def _smooth_outer_side(cfg, center, angle_deg, beta=15.0):
    """Smooth outer side length using log-sum-exp."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    cx, cy = center
    vals = []
    for (px, py, pangle) in cfg:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                vals.append(val)
    vals = np.array(vals)
    max_val = np.max(vals)
    shifted = vals - max_val
    lse = max_val + (1.0 / beta) * np.log(np.sum(np.exp(beta * shifted)))
    return lse

def _smooth_collision_penalty(all_verts, beta=20.0):
    """Smooth collision penalty."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _min_overlap_depth(all_verts[i], all_verts[j])
            if depth > 0:
                penalty += depth * depth
    return penalty

def _golden_section_minimize(f, a, b, tol=1e-7, max_iter=200):
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
    return (a + b) / 2.0

def _optimize_outer_rotation(cfg, center, time_budget=5.0):
    """Optimize outer hexagon rotation and center via composite body approach."""
    t_start = time.time()
    all_verts = []
    for (px, py, pangle) in cfg:
        verts = _hex_vertices(px, py, pangle)
        all_verts.extend(verts)
    all_verts_np = np.array(all_verts)
    
    cx_body = float(all_verts_np[:, 0].mean())
    cy_body = float(all_verts_np[:, 1].mean())
    apothem_unit = math.sqrt(3) / 2.0
    
    def side_for_rotation(angle_deg, cx, cy):
        a = math.radians(angle_deg)
        normals = np.array([
            [math.cos(a + math.radians(90 + 60 * k)),
             math.sin(a + math.radians(90 + 60 * k))]
            for k in range(6)
        ])
        proj = (all_verts_np - np.array([cx, cy])) @ normals
        return float(np.max(proj) / apothem_unit)
    
    def _chebyshev_center(normals, all_verts_np, init_cx, init_cy, max_iter=50, tol=1e-12):
        n_c = len(normals) * len(all_verts_np)
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = np.zeros((n_c, 3))
        b_ub = np.zeros(n_c)
        idx = 0
        for (nx, ny) in normals:
            for (vx, vy) in all_verts_np:
                A_ub[idx, 0] = -nx
                A_ub[idx, 1] = -ny
                A_ub[idx, 2] = -apothem_unit
                b_ub[idx] = -(nx * vx + ny * vy)
                idx += 1
        bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0.0, np.inf)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return float(res.x[0]), float(res.x[1])
        except Exception:
            pass
        # Fallback iterative
        cx, cy = init_cx, init_cy
        for it in range(max_iter):
            max_val = 0.0
            active_sum_x, active_sum_y, count = 0.0, 0.0, 0
            for (nx, ny) in normals:
                for (vx, vy) in all_verts_np:
                    val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                    if val > max_val:
                        max_val = val
            threshold = max_val * 0.95
            for (nx, ny) in normals:
                for (vx, vy) in all_verts_np:
                    val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                    if val >= threshold:
                        active_sum_x += vx
                        active_sum_y += vy
                        count += 1
            if count > 0:
                new_cx = active_sum_x / count
                new_cy = active_sum_y / count
                cx += 0.5 * (new_cx - cx)
                cy += 0.5 * (new_cy - cy)
            if abs(cx - init_cx) < tol and abs(cy - init_cy) < tol:
                break
            init_cx, init_cy = cx, cy
        return cx, cy
    
    def side_for_rotation_opt(angle_deg):
        a = math.radians(angle_deg)
        normals = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        cx, cy = _chebyshev_center(normals, all_verts_np, cx_body, cy_body, max_iter=30, tol=1e-11)
        return side_for_rotation(angle_deg, cx, cy), (cx, cy)
    
    def f_rot(theta):
        s, _ = side_for_rotation_opt(theta)
        return s
    
    best_angle = _golden_section_minimize(f_rot, 0.0, 60.0, tol=1e-7, max_iter=80)
    best_side, best_center = side_for_rotation_opt(best_angle)
    
    for da in [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5]:
        trial_angle = (best_angle + da) % 60.0
        trial_side, trial_center = side_for_rotation_opt(trial_angle)
        if trial_side < best_side:
            best_side = trial_side
            best_center = trial_center
            best_angle = trial_angle
    
    return best_center, best_angle, best_side

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def full_score(cfg, center, angle):
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, tuple(center), angle)
    
    def make_hex_lattice():
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts
    
    def make_ring():
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts
    
    def make_flower():
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2 * math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_tight_hex12():
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2.0 * math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_compact_grid():
        pts = []
        for i in range(6):
            x = i * 1.86
            y = 0.0
            pts.append([x - 2.79, y, 0.0])
        for i in range(6):
            x = i * 1.86 + 0.93
            y = math.sqrt(3) * 0.95
            pts.append([x - 2.79, y, 0.0])
        return pts
    
    def smooth_objective(v, beta=15.0):
        cfg = []
        for i in range(N):
            cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
        center = (v[3*N], v[3*N+1])
        outer_angle = v[3*N+2]
        
        all_verts = []
        for p in cfg:
            verts = _hex_vertices(p[0], p[1], p[2])
            all_verts.append(verts)
        
        smooth_side = _smooth_outer_side(cfg, center, outer_angle, beta=beta)
        collision_penalty = _smooth_collision_penalty(all_verts, beta=20.0)
        
        disp_penalty = 0.0
        for i in range(N):
            disp_penalty += v[3*i]**2 + v[3*i+1]**2
        disp_penalty += v[3*N]**2 + v[3*N+1]**2
        disp_penalty *= 0.01
        
        return smooth_side + 50.0 * collision_penalty + disp_penalty
    
    def gradient_descent_polish(cfg_init, center_init, angle_init, time_budget, beta=15.0):
        def flatten(cfg, center, angle):
            v = []
            for p in cfg:
                v.extend([p[0], p[1], p[2]])
            v.extend([center[0], center[1], angle])
            return np.array(v, dtype=float)
        
        def unflatten(v):
            cfg = []
            for i in range(N):
                cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
            center = [v[3*N], v[3*N+1]]
            angle = v[3*N+2]
            return cfg, center, angle
        
        def obj(v):
            if time.time() - t0 > TIME_LIMIT - 2.0:
                return 1e6
            return smooth_objective(v, beta=beta)
        
        v = flatten(cfg_init, center_init, angle_init)
        
        result = minimize(
            obj, v, method='L-BFGS-B',
            options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        best_v = v.copy()
        best_f = smooth_objective(v, beta=beta)
        
        if result.fun < best_f:
            best_v = result.x.copy()
            best_f = result.fun
        
        return best_v, best_f
    
    # Try multiple initial configurations
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    inits = [make_hex_lattice(), make_ring(), make_flower(), make_tight_hex12(), make_compact_grid()]
    for init in inits:
        s = full_score(init, (0.0, 0.0), 0.0)
        if s is not None and s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0
    
    if best_cfg is None:
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0
    
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score
    
    # Simulated annealing phase
    T = 0.8
    T_end = 0.002
    alpha = 0.9996
    sa_time_limit = TIME_LIMIT - 40.0
    
    lattice_positions = []
    row_height = math.sqrt(3)
    col_width = 2.0
    for r in range(5):
        for c in range(6):
            x = c * col_width + (col_width / 2 if r % 2 else 0.0)
            y = r * row_height
            lattice_positions.append((x - 3.0, y - 2.0 * row_height))
    
    def _compute_polys(cfg):
        return [_hex_vertices(p[0], p[1], p[2]) for p in cfg]

    def _quick_valid(polys, tol=1e-9):
        m = len(polys)
        # Quick center-distance prefilter
        centers = np.array([[sum(v[0] for v in p)/6, sum(v[1] for v in p)/6] for p in polys])
        for i in range(m):
            for j in range(i + 1, m):
                if math.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1]) > 4.0:
                    continue
                if _separating_axis_check(polys[i], polys[j], tol):
                    return False
        return True

    def _quick_outer_side(cfg, center, angle_deg):
        a = math.radians(angle_deg)
        apothem_unit = math.sqrt(3) / 2.0
        normals = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        cx, cy = center
        max_need = 0.0
        for p in cfg:
            verts = _hex_vertices(p[0], p[1], p[2])
            for (vx, vy) in verts:
                for (nx, ny) in normals:
                    val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                    if val > max_need:
                        max_need = val
        return max_need

    while time.time() - t0 < sa_time_limit:
        r = random.random()
        move_type = None
        move_data = None

        # Scale step sizes with temperature for better exploration/exploitation
        pos_scale = 0.25 * (0.5 + T)
        ang_scale = 6.0 * (0.3 + T)
        ctr_scale = 0.12 * (0.5 + T)
        oang_scale = 2.5 * (0.3 + T)
        pair_scale = 0.18 * (0.5 + T)

        if r < 0.45:
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = pos_scale
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
            move_type = 'pos'
            move_data = (idx, old_x, old_y)
        elif r < 0.60:
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, ang_scale)
            move_type = 'ang'
            move_data = (idx, old_ang)
        elif r < 0.70:
            old_cx, old_cy = center[0], center[1]
            step = ctr_scale
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
            move_type = 'center'
            move_data = (old_cx, old_cy)
        elif r < 0.80:
            old_angle = angle
            angle += random.gauss(0, oang_scale)
            move_type = 'outer_ang'
            move_data = old_angle
        elif r < 0.90:
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = pair_scale
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
            move_type = 'pair'
            move_data = (idx1, old_x1, old_y1, idx2, old_x2, old_y2)
        else:
            # Rotate a pair of hexagons together (preserves relative geometry)
            idx1, idx2 = random.sample(range(N), 2)
            old_a1, old_a2 = cfg[idx1][2], cfg[idx2][2]
            da = random.gauss(0, ang_scale * 0.5)
            cfg[idx1][2] += da
            cfg[idx2][2] += da
            move_type = 'ang'
            move_data = (idx1, old_a1, idx2, old_a2)

        polys = _compute_polys(cfg)
        valid = _quick_valid(polys)

        if valid:
            new_score = _quick_outer_side(cfg, tuple(center), angle)
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                if move_type == 'pos':
                    cfg[move_data[0]][0], cfg[move_data[0]][1] = move_data[1], move_data[2]
                elif move_type == 'ang':
                    if len(move_data) == 2:
                        cfg[move_data[0]][2] = move_data[1]
                    else:
                        cfg[move_data[0]][2] = move_data[1]
                        cfg[move_data[2]][2] = move_data[3]
                elif move_type == 'center':
                    center[0], center[1] = move_data[0], move_data[1]
                elif move_type == 'outer_ang':
                    angle = move_data
                elif move_type == 'pair':
                    cfg[move_data[0]][0], cfg[move_data[0]][1] = move_data[1], move_data[2]
                    cfg[move_data[3]][0], cfg[move_data[3]][1] = move_data[4], move_data[5]
        else:
            if move_type == 'pos':
                cfg[move_data[0]][0], cfg[move_data[0]][1] = move_data[1], move_data[2]
            elif move_type == 'ang':
                if len(move_data) == 2:
                    cfg[move_data[0]][2] = move_data[1]
                else:
                    cfg[move_data[0]][2] = move_data[1]
                    cfg[move_data[2]][2] = move_data[3]
            elif move_type == 'center':
                center[0], center[1] = move_data[0], move_data[1]
            elif move_type == 'outer_ang':
                angle = move_data
            elif move_type == 'pair':
                cfg[move_data[0]][0], cfg[move_data[0]][1] = move_data[1], move_data[2]
                cfg[move_data[3]][0], cfg[move_data[3]][1] = move_data[4], move_data[5]

        T = max(T_end, T * alpha)
    
    # Composite body optimization
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 10.0:
        try:
            cb_center, cb_angle, cb_side = _optimize_outer_rotation(
                best_cfg, best_center, time_budget=min(remaining_time * 0.25, 6.0)
            )
            if cb_side < best_score:
                if _is_valid(best_cfg):
                    best_score = cb_side
                    best_center = cb_center
                    best_angle = cb_angle
        except Exception:
            pass
    
    # Multi-stage gradient descent polish with increasing smoothness
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 12.0:
        try:
            polished_v, polished_f = gradient_descent_polish(
                best_cfg, best_center, best_angle, min(remaining_time * 0.35, 12.0), beta=10.0
            )
            cfg_polished = []
            for i in range(N):
                cfg_polished.append([polished_v[3*i], polished_v[3*i+1], polished_v[3*i+2]])
            center_polished = [polished_v[3*N], polished_v[3*N+1]]
            angle_polished = polished_v[3*N+2]
            
            polished_score = full_score(cfg_polished, tuple(center_polished), angle_polished)
            if polished_score is not None and polished_score < best_score:
                best_score = polished_score
                best_cfg = cfg_polished
                best_center = center_polished
                best_angle = angle_polished
        except Exception:
            pass
    
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 8.0:
        try:
            polished_v2, polished_f2 = gradient_descent_polish(
                best_cfg, best_center, best_angle, min(remaining_time * 0.4, 10.0), beta=25.0
            )
            cfg_polished2 = []
            for i in range(N):
                cfg_polished2.append([polished_v2[3*i], polished_v2[3*i+1], polished_v2[3*i+2]])
            center_polished2 = [polished_v2[3*N], polished_v2[3*N+1]]
            angle_polished2 = polished_v2[3*N+2]
            
            polished_score2 = full_score(cfg_polished2, tuple(center_polished2), angle_polished2)
            if polished_score2 is not None and polished_score2 < best_score:
                best_score = polished_score2
                best_cfg = cfg_polished2
                best_center = center_polished2
                best_angle = angle_polished2
        except Exception:
            pass
    
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 6.0:
        try:
            polished_v3, polished_f3 = gradient_descent_polish(
                best_cfg, best_center, best_angle, min(remaining_time * 0.4, 8.0), beta=50.0
            )
            cfg_polished3 = []
            for i in range(N):
                cfg_polished3.append([polished_v3[3*i], polished_v3[3*i+1], polished_v3[3*i+2]])
            center_polished3 = [polished_v3[3*N], polished_v3[3*N+1]]
            angle_polished3 = polished_v3[3*N+2]
            
            polished_score3 = full_score(cfg_polished3, tuple(center_polished3), angle_polished3)
            if polished_score3 is not None and polished_score3 < best_score:
                best_score = polished_score3
                best_cfg = cfg_polished3
                best_center = center_polished3
                best_angle = angle_polished3
        except Exception:
            pass
    
    # Final composite body optimization
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 6.0:
        try:
            cb_center2, cb_angle2, cb_side2 = _optimize_outer_rotation(
                best_cfg, best_center, time_budget=min(remaining_time * 0.3, 4.0)
            )
            if cb_side2 < best_score:
                if _is_valid(best_cfg):
                    best_score = cb_side2
                    best_center = cb_center2
                    best_angle = cb_angle2
        except Exception:
            pass
        
        remaining_time = TIME_LIMIT - (time.time() - t0)
        if remaining_time > 4.0:
            try:
                all_verts_final = []
                for (px, py, pangle) in best_cfg:
                    verts = _hex_vertices(px, py, pangle)
                    all_verts_final.extend(verts)
                all_verts_final = np.array(all_verts_final)
                apothem_unit = math.sqrt(3) / 2.0
                for dangle in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
                    trial_angle = (best_angle + dangle) % 60.0
                    a = math.radians(trial_angle)
                    normals = np.array([
                        [math.cos(a + math.radians(90 + 60 * k)),
                         math.sin(a + math.radians(90 + 60 * k))]
                        for k in range(6)
                    ])
                    cx, cy = best_center[0], best_center[1]
                    for _ in range(30):
                        proj = (all_verts_final - np.array([cx, cy])) @ normals
                        max_val = float(np.max(proj)) / apothem_unit
                        active_mask = proj > max_val * apothem_unit * 0.95
                        if np.any(active_mask):
                            active_verts = all_verts_final[active_mask]
                            cx += 0.3 * (active_verts[:, 0].mean() - cx)
                            cy += 0.3 * (active_verts[:, 1].mean() - cy)
                    trial_side = 0.0
                    proj = (all_verts_final - np.array([cx, cy])) @ normals
                    trial_side = float(np.max(proj)) / apothem_unit
                    if trial_side < best_score:
                        if _is_valid(best_cfg):
                            best_score = trial_side
                            best_center = [cx, cy]
                            best_angle = trial_angle
            except Exception:
                pass
    
    # Final high-precision polish: Nelder-Mead on composite body
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 8.0:
        try:
            all_verts_fixed = []
            for (px, py, pangle) in best_cfg:
                verts = _hex_vertices(px, py, pangle)
                all_verts_fixed.extend(verts)
            all_verts_fixed = np.array(all_verts_fixed)
            apothem_unit = math.sqrt(3) / 2.0

            def composite_obj(params):
                cx, cy, oa = params[0], params[1], params[2]
                a = math.radians(oa)
                normals = np.array([
                    [math.cos(a + math.radians(90 + 60 * k)),
                     math.sin(a + math.radians(90 + 60 * k))]
                    for k in range(6)
                ])
                proj = (all_verts_fixed - np.array([cx, cy])) @ normals
                return float(np.max(proj) / apothem_unit)

            x0 = np.array([best_center[0], best_center[1], best_angle])
            res = minimize(
                composite_obj, x0, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True}
            )
            if res.fun < best_score:
                if _is_valid(best_cfg):
                    best_score = res.fun
                    best_center = [float(res.x[0]), float(res.x[1])]
                    best_angle = float(res.x[2]) % 60.0
        except Exception:
            pass

    # Ultra-fine final polish: micro-perturbations with composite body re-optimization
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 5.0:
        try:
            apothem_unit = math.sqrt(3) / 2.0
            max_iters = 150
            for _ in range(max_iters):
                if time.time() - t0 > TIME_LIMIT - 1.5:
                    break
                cfg_try = [row[:] for row in best_cfg]
                # Perturb 1-2 hexagons slightly
                n_perturb = random.choice([1, 1, 1, 2])
                for _ in range(n_perturb):
                    idx = random.randrange(N)
                    cfg_try[idx][0] += random.gauss(0, 0.008)
                    cfg_try[idx][1] += random.gauss(0, 0.008)
                    if random.random() < 0.3:
                        cfg_try[idx][2] += random.gauss(0, 0.5)
                if not _is_valid(cfg_try):
                    continue
                all_v = []
                for p in cfg_try:
                    all_v.extend(_hex_vertices(p[0], p[1], p[2]))
                all_v = np.array(all_v)
                # Quick composite body optimization
                best_s = best_score
                best_cx, best_cy, best_oa = best_center[0], best_center[1], best_angle
                for _nm in range(100):
                    if time.time() - t0 > TIME_LIMIT - 1.5:
                        break
                    a = math.radians(best_oa)
                    normals = np.array([
                        [math.cos(a + math.radians(90 + 60 * k)),
                         math.sin(a + math.radians(90 + 60 * k))]
                        for k in range(6)
                    ])
                    proj = (all_v - np.array([best_cx, best_cy])) @ normals
                    s = float(np.max(proj)) / apothem_unit
                    if s < best_s:
                        best_s = s
                    # Move center toward active vertices
                    active_mask = proj > s * apothem_unit * 0.97
                    if np.any(active_mask):
                        active_verts = all_v[active_mask]
                        best_cx += 0.15 * (active_verts[:, 0].mean() - best_cx)
                        best_cy += 0.15 * (active_verts[:, 1].mean() - best_cy)
                    # Small angle nudge
                    best_oa += random.gauss(0, 0.03)
                if best_s < best_score:
                    best_score = best_s
                    best_cfg = cfg_try
                    best_center = [best_cx, best_cy]
                    best_angle = best_oa % 60.0
        except Exception:
            pass
    
    # Final quick refinement
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 2.5:
        for _ in range(200):
            if time.time() - t0 > TIME_LIMIT - 1.0:
                break
            cfg_try = [row[:] for row in best_cfg]
            center_try = best_center[:]
            angle_try = best_angle
            idx = random.randrange(N)
            cfg_try[idx][0] += random.gauss(0, 0.015)
            cfg_try[idx][1] += random.gauss(0, 0.015)
            score_try = full_score(cfg_try, tuple(center_try), angle_try)
            if score_try is not None and score_try < best_score:
                best_score = score_try
                best_cfg = cfg_try
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
