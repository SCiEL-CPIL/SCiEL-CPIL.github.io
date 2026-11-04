import math
import time
import random
import numpy as np
from scipy.optimize import minimize

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

def _hex_vertices_np(x, y, rot_deg):
    """Vectorized hex vertices for a single hexagon."""
    base = np.radians([0, 60, 120, 180, 240, 300])
    a = np.radians(rot_deg)
    angles = base + a
    return np.column_stack((x + np.cos(angles), y + np.sin(angles)))

def _hex_vertices_batch(params):
    """params: (N, 3) array of [x, y, rot_deg]. Returns (N, 6, 2)."""
    N_pts = len(params)
    base = np.radians([0, 60, 120, 180, 240, 300])
    rots = np.radians(params[:, 2])
    angles = base[None, :] + rots[:, None]
    x = params[:, 0:1] + np.cos(angles)
    y = params[:, 1:2] + np.sin(angles)
    return np.stack((x, y), axis=-1)

def _sat_overlap_depth(v1, v2):
    """Returns penetration depth (positive if overlapping, negative if separated)."""
    axes = []
    for v in (v1, v2):
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        proj2 = v2 @ n
        if np.max(proj1) <= np.min(proj2) + 1e-9 or np.max(proj2) <= np.min(proj1) + 1e-9:
            return -1.0  # separated
        overlap = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
        min_overlap = min(min_overlap, overlap)
    
    return min_overlap

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    verts_list = [_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]) for i in range(n)]
    
    axes_list = []
    for i in range(n):
        axes = []
        vi = verts_list[i]
        for k in range(6):
            x1, y1 = vi[k]
            x2, y2 = vi[(k + 1) % 6]
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < 1e-12:
                continue
            axes.append((-dy / length, dx / length))
        axes_list.append(axes)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq > 12.0:  # (2*sqrt(3))^2
                continue
            if dist_sq < 3.0 - 1e-9:  # sqrt(3)^2
                return False
            
            vi = verts_list[i]
            vj = verts_list[j]
            separated = False
            for axis in axes_list[i]:
                ax, ay = axis
                minA = min(vx * ax + vy * ay for vx, vy in vi)
                maxA = max(vx * ax + vy * ay for vx, vy in vi)
                minB = min(vx * ax + vy * ay for vx, vy in vj)
                maxB = max(vx * ax + vy * ay for vx, vy in vj)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    separated = True
                    break
            if separated:
                continue
            for axis in axes_list[j]:
                ax, ay = axis
                minA = min(vx * ax + vy * ay for vx, vy in vi)
                maxA = max(vx * ax + vy * ay for vx, vy in vi)
                minB = min(vx * ax + vy * ay for vx, vy in vj)
                maxB = max(vx * ax + vy * ay for vx, vy in vj)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    separated = True
                    break
            if not separated:
                return False
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _make_hex_grid(rows, cols, gap=0.0):
    pts = []
    sqrt3 = math.sqrt(3)
    spacing_x = sqrt3 + gap
    spacing_y = 1.5 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, 0.0])
    # Center the configuration
    if pts:
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        for p in pts:
            p[0] -= mx
            p[1] -= my
    return pts[:N]

def _make_flower(gap=0.0):
    pts = []
    pts.append([0.0, 0.0, 0.0])
    r1 = math.sqrt(3) + gap
    for i in range(6):
        angle = math.radians(60 * i)
        pts.append([r1 * math.cos(angle), r1 * math.sin(angle), 0.0])
    r2 = 2 * math.sqrt(3) + gap
    for i in range(5):
        angle = math.radians(60 * i + 30)
        pts.append([r2 * math.cos(angle), r2 * math.sin(angle), 0.0])
    return pts[:N]

def _make_random():
    pts = []
    for _ in range(N):
        pts.append([random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0, 60)])
    return pts

def _smooth_objective(params, outer_rot=0.0):
    """Objective: outer hex side + smooth collision penalty."""
    xys_rots = params[:3*N].reshape(N, 3)
    cx_out = params[3*N]
    cy_out = params[3*N + 1]
    
    verts = _hex_vertices_batch(xys_rots)
    
    # Compute outer hex side
    a = math.radians(outer_rot)
    apothem = math.sqrt(3) / 2.0
    normals = np.array([[math.cos(a + math.radians(90 + 60*k)), 
                         math.sin(a + math.radians(90 + 60*k))] for k in range(6)])
    
    all_pts = verts.reshape(-1, 2)
    rel_pts = all_pts - np.array([cx_out, cy_out])
    projs = rel_pts @ normals.T
    max_proj = np.max(np.abs(projs))
    L = 2.0 / math.sqrt(3.0) * max_proj
    
    # Smooth collision penalty using SDF-like approach
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xys_rots[i, 0] - xys_rots[j, 0]
            dy = xys_rots[i, 1] - xys_rots[j, 1]
            dist = math.hypot(dx, dy)
            if dist < math.sqrt(3):
                overlap = math.sqrt(3) - dist
                penalty += overlap * overlap * 50.0
    
    return L + penalty

def construct():
    t0 = time.time()
    random.seed(42)
    
    def time_left():
        return TIME_LIMIT - (time.time() - t0)
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Generate diverse initial configurations
    initializers = [
        lambda: _make_hex_grid(4, 3, gap=0.1),
        lambda: _make_hex_grid(3, 4, gap=0.1),
        lambda: _make_hex_grid(2, 6, gap=0.1),
        lambda: _make_hex_grid(6, 2, gap=0.1),
        lambda: _make_flower(gap=0.1),
        lambda: _make_random(),
    ]
    
    outer_rots = [0.0, 30.0, 15.0]
    
    # Phase 0: Evaluate initial configurations
    for init_func in initializers:
        if time_left() < 60:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            for orot in outer_rots:
                side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), orot)
                if side < best_side:
                    best_side = side
                    best_cfg = [row[:] for row in cfg]
                    best_center = (0.0, 0.0)
                    best_angle = orot
    
    if best_cfg is None:
        cfg = _make_hex_grid(3, 4, gap=0.2)
        best_cfg = cfg
        best_side = _outer_hex_side(np.array([p[0] for p in cfg]), 
                                     np.array([p[1] for p in cfg]),
                                     np.array([p[2] for p in cfg]), (0, 0), 0)
    
    # Phase 1: Multi-start L-BFGS-B optimization
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    def _objective(params):
        if time_left() < 5:
            return 1e6
        return _smooth_objective(params, best_angle)
    
    def _gradient(params):
        if time_left() < 5:
            return np.zeros(len(params))
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros(len(params))
        f0 = _objective(params)
        for i in range(len(params)):
            params[i] += eps
            f1 = _objective(params)
            params[i] -= eps
            grad[i] = (f1 - f0) / eps
        return grad
    
    # Try multiple starting points
    start_configs = [best_cfg]
    for init_func in initializers[:3]:
        if time_left() < 40:
            break
        cfg = init_func()
        if len(cfg) == N and _is_valid_fast(np.array([p[0] for p in cfg]), 
                                             np.array([p[1] for p in cfg]),
                                             np.array([p[2] for p in cfg])):
            start_configs.append(cfg)
    
    for cfg in start_configs:
        if time_left() < 15:
            break
        
        init_params = np.zeros(3 * N + 2)
        for i in range(N):
            init_params[i] = cfg[i][0]
            init_params[N + i] = cfg[i][1]
            init_params[2 * N + i] = cfg[i][2]
        init_params[3 * N] = best_center[0]
        init_params[3 * N + 1] = best_center[1]
        
        try:
            result = minimize(
                _objective,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_side:
                opt_params = result.x
                xs_opt = opt_params[0:N]
                ys_opt = opt_params[N:2*N]
                angles_opt = opt_params[2*N:3*N]
                cx_out_opt = opt_params[3*N]
                cy_out_opt = opt_params[3*N + 1]
                
                if _is_valid_fast(xs_opt, ys_opt, angles_opt):
                    side_opt = _outer_hex_side(xs_opt, ys_opt, angles_opt, 
                                               (cx_out_opt, cy_out_opt), best_angle)
                    if side_opt < best_side:
                        best_side = side_opt
                        best_cfg = [[xs_opt[i], ys_opt[i], angles_opt[i]] for i in range(N)]
                        best_center = (cx_out_opt, cy_out_opt)
        except Exception:
            pass
    
    # Phase 2: Simulated Annealing refinement
    if time_left() > 30:
        cfg = [row[:] for row in best_cfg]
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        cur_score = best_side
        T = 0.2
        T_end = 0.001
        alpha = 0.997
        
        apothem = math.sqrt(3) / 2.0
        normals = []
        a = math.radians(best_angle)
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        def _quick_score(cx_a, cy_a, ang_a):
            max_need = 0.0
            for i in range(N):
                aa = math.radians(ang_a[i])
                ca, sa = math.cos(aa), math.sin(aa)
                for k in range(6):
                    theta = math.radians(60 * k)
                    vx = cx_a[i] + ca * math.cos(theta) - sa * math.sin(theta)
                    vy = cy_a[i] + sa * math.cos(theta) + ca * math.sin(theta)
                    for (nx, ny) in normals:
                        val = (nx * (vx - best_center[0]) + ny * (vy - best_center[1])) / apothem
                        if val > max_need:
                            max_need = val
            return max_need
        
        while time_left() > 10:
            idx = random.randrange(N)
            old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
            
            cx_arr[idx] += random.gauss(0, 0.08)
            cy_arr[idx] += random.gauss(0, 0.08)
            angle_arr[idx] += random.gauss(0, 2.0)
            
            if _is_valid_fast(cx_arr, cy_arr, angle_arr):
                new_score = _quick_score(cx_arr, cy_arr, angle_arr)
                if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                    cur_score = new_score
                    if new_score < best_side:
                        best_side = new_score
                        best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
                else:
                    cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
            
            T = max(T_end, T * alpha)
    
    # Phase 3: Final outer hexagon optimization
    if time_left() > 5:
        cx_final = np.array([p[0] for p in best_cfg])
        cy_final = np.array([p[1] for p in best_cfg])
        angle_final = np.array([p[2] for p in best_cfg])
        
        verts_best = [_hex_vertices(cx_final[i], cy_final[i], angle_final[i]) for i in range(N)]
        
        for outer_angle in range(0, 30, 3):
            if time_left() < 2:
                break
            
            a = math.radians(outer_angle)
            apothem = math.sqrt(3) / 2.0
            normals = []
            for k in range(6):
                na = a + math.radians(90 + 60 * k)
                normals.append((math.cos(na), math.sin(na)))
            
            cx_c, cy_c = best_center[0], best_center[1]
            for _iter in range(15):
                max_proj = 0.0
                for i in range(N):
                    for (vx, vy) in verts_best[i]:
                        for (nx, ny) in normals:
                            val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem
                            if val > max_proj:
                                max_proj = val
                
                best_cx, best_cy, best_val = cx_c, cy_c, max_proj
                for dx in [-0.15, -0.05, 0, 0.05, 0.15]:
                    for dy in [-0.15, -0.05, 0, 0.05, 0.15]:
                        ncx, ncy = cx_c + dx, cy_c + dy
                        val = 0.0
                        for i in range(N):
                            for (vx, vy) in verts_best[i]:
                                for (nx, ny) in normals:
                                    v = (nx * (vx - ncx) + ny * (vy - ncy)) / apothem
                                    if v > val:
                                        val = v
                        if val < best_val:
                            best_val = val
                            best_cx, best_cy = ncx, ncy
                
                if abs(best_cx - cx_c) < 1e-4 and abs(best_cy - cy_c) < 1e-4:
                    break
                cx_c, cy_c = best_cx, best_cy
            
            if max_proj < best_side:
                best_side = max_proj
                best_center = (cx_c, cy_c)
                best_angle = float(outer_angle)
    
    # Final verification
    cx_f = np.array([p[0] for p in best_cfg])
    cy_f = np.array([p[1] for p in best_cfg])
    angle_f = np.array([p[2] for p in best_cfg])
    
    if not _is_valid_fast(cx_f, cy_f, angle_f):
        cfg = _make_hex_grid(3, 4, gap=0.3)
        best_cfg = cfg
        best_side = _outer_hex_side(np.array([p[0] for p in cfg]),
                                     np.array([p[1] for p in cfg]),
                                     np.array([p[2] for p in cfg]), (0, 0), 0)
        best_center = (0.0, 0.0)
        best_angle = 0.0
    else:
        best_side = _outer_hex_side(cx_f, cy_f, angle_f, best_center, best_angle)
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, best_center, best_side, best_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
