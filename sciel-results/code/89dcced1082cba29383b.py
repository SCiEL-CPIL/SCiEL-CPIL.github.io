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

def _min_separation(vertsA, vertsB):
    """Returns the minimum distance between two convex polygons.
    If they overlap, returns a negative value representing the penetration depth
    (approximated by the minimum overlap width on the most critical axis)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    max_gap = -float('inf')
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        # Gap is distance between max of one and min of other
        # If A is to the left of B: minB - maxA
        # If B is to the left of A: minA - maxB
        gap = min(minB - maxA, minA - maxB)
        if gap > max_gap:
            max_gap = gap
    # If max_gap > 0, they are separated by max_gap.
    # If max_gap < 0, they overlap. The penetration depth is -max_gap.
    return max_gap

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    for i in range(n):
        xi, yi = cx_arr[i], cy_arr[i]
        vi = verts_list[i]
        for j in range(i + 1, n):
            xj, yj = cx_arr[j], cy_arr[j]
            vj = verts_list[j]
            
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            
            if dist_sq < 4.0 - 1e-9:
                return False
            
            sep = _min_separation(vi, vj)
            if sep < -1e-9:
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

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_flower_initial():
        pts = []
        pts.append([0.0, 0.0, 0.0])
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        r2 = r1 * 1.732
        angles = [30, 90, 150, 210, 270]
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y, 0.0])
        return pts[:12]
    
    def make_lattice_initial():
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        for c in range(3):
            x = c * col_width - col_width
            y = -row_height
            pts.append([x, y, 0.0])
        for c in range(4):
            x = c * col_width - col_width - col_width/2
            y = 0.0
            pts.append([x, y, 0.0])
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        pts.append([-col_width*2 - col_width/2 - col_width, 0.0, 0.0])
        pts.append([col_width*2 - col_width/2 + col_width, 0.0, 0.0])
        return pts

    initializers = [
        make_flower_initial,
        make_lattice_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 60:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
                best_center = (0.0, 0.0)
                best_angle = 0.0
    
    if best_cfg is None:
        cfg = make_lattice_initial()
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # --- Phase 1: Simulated Annealing ---
    cfg = [row[:] for row in best_cfg]
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    cur_score = best_side
    T = 0.3
    T_end = 0.001
    alpha = 0.995
    
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    def _quick_score(cx_arr, cy_arr, angle_arr):
        max_need = 0.0
        for i in range(N):
            a = math.radians(angle_arr[i])
            ca, sa = math.cos(a), math.sin(a)
            for k in range(6):
                theta = math.radians(60 * k)
                vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
                vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
                for (nx, ny) in normals:
                    val = (nx * vx + ny * vy) / apothem_unit
                    if val > max_need:
                        max_need = val
        return max_need

    sa_restarts = 3
    steps_per_restart = 3000
    
    for restart in range(sa_restarts):
        if time.time() - t0 > TIME_LIMIT - 40:
            break
        
        if restart > 0:
            cx_arr[:] = [p[0] for p in best_cfg]
            cy_arr[:] = [p[1] for p in best_cfg]
            angle_arr[:] = [p[2] for p in best_cfg]
            cur_score = best_side
            T = 0.2
        
        for step in range(steps_per_restart):
            if time.time() - t0 > TIME_LIMIT - 40:
                break
            
            r = random.random()
            idx = random.randrange(N)
            old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
            idx2 = -1
            old_x2, old_y2, old_a2 = 0, 0, 0
            
            if r < 0.65:
                cx_arr[idx] += random.gauss(0, 0.10)
                cy_arr[idx] += random.gauss(0, 0.10)
                angle_arr[idx] += random.gauss(0, 2.0)
            elif r < 0.80:
                angle_arr[idx] += random.gauss(0, 5.0)
            elif r < 0.90:
                idx2 = random.randrange(N)
                if idx2 == idx:
                    idx2 = (idx + 1) % N
                old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
                cx_arr[idx], cx_arr[idx2] = cx_arr[idx2], cx_arr[idx]
                cy_arr[idx], cy_arr[idx2] = cy_arr[idx2], cy_arr[idx]
                angle_arr[idx], angle_arr[idx2] = angle_arr[idx2], angle_arr[idx]
            else:
                idx2 = random.randrange(N)
                if idx2 == idx:
                    idx2 = (idx + 1) % N
                old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
                cx_arr[idx] += random.gauss(0, 0.06)
                cy_arr[idx] += random.gauss(0, 0.06)
                cx_arr[idx2] += random.gauss(0, 0.06)
                cy_arr[idx2] += random.gauss(0, 0.06)
                angle_arr[idx] += random.gauss(0, 1.0)
                angle_arr[idx2] += random.gauss(0, 1.0)
            
            valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
            
            if valid:
                new_score = _quick_score(cx_arr, cy_arr, angle_arr)
                if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                    cur_score = new_score
                    if new_score < best_side:
                        best_side = new_score
                        best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
                        best_center = (0.0, 0.0)
                        best_angle = 0.0
                else:
                    cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                    if idx2 != -1:
                        cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                if idx2 != -1:
                    cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
            
            T = max(T_end, T * alpha)

    # --- Phase 2: L-BFGS-B Polishing ---
    # Objective: Minimize Outer Side + Penalty for Overlaps
    def objective(params):
        cx_arr = params[0:N]
        cy_arr = params[N:2*N]
        angle_arr = params[2*N:3*N]
        
        # Calculate outer side
        side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
        
        # Calculate overlap penalty
        penalty = 0.0
        verts_list = [_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                # Quick distance check
                dx = cx_arr[i] - cx_arr[j]
                dy = cy_arr[i] - cy_arr[j]
                if dx*dx + dy*dy > (2.2)**2:
                    continue
                sep = _min_separation(verts_list[i], verts_list[j])
                if sep < 0:
                    penalty += (sep * sep) * 100.0
        
        return side + penalty

    # Initial guess from best SA result
    x0 = []
    for p in best_cfg:
        x0.append(p[0])
    for p in best_cfg:
        x0.append(p[1])
    for p in best_cfg:
        x0.append(p[2])
    
    x0 = np.array(x0)
    
    # Run L-BFGS-B
    if time.time() - t0 < TIME_LIMIT - 10:
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 500})
            if res.success or res.fun < best_side:
                new_params = res.x
                new_cx = new_params[0:N]
                new_cy = new_params[N:2*N]
                new_angle = new_params[2*N:3*N]
                
                # Verify validity
                if _is_valid_fast(new_cx, new_cy, new_angle):
                    new_side = _outer_hex_side(new_cx, new_cy, new_angle, (0.0, 0.0), 0.0)
                    if new_side < best_side:
                        best_side = new_side
                        best_cfg = [[new_cx[i], new_cy[i], new_angle[i]] for i in range(N)]
                        best_center = (0.0, 0.0)
                        best_angle = 0.0
        except Exception:
            pass

    # --- Phase 3: Optimize outer hexagon center and angle ---
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = best_center
    best_final_angle = best_angle
    
    verts_best = [_hex_vertices(cx_best[i], cy_best[i], angle_best[i]) for i in range(N)]
    
    for outer_angle in range(0, 30, 1):
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        
        a = math.radians(outer_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals_out = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals_out.append((math.cos(na), math.sin(na)))
        
        cx_c, cy_c = best_final_center[0], best_final_center[1]
        
        for _iter in range(30):
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_best[i]:
                    for (nx, ny) in normals_out:
                        val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                        if val > max_proj:
                            max_proj = val
            
            best_cx, best_cy, best_val = cx_c, cy_c, max_proj
            
            for dx in [-0.15, -0.08, 0, 0.08, 0.15]:
                for dy in [-0.15, -0.08, 0, 0.08, 0.15]:
                    ncx, ncy = cx_c + dx, cy_c + dy
                    val = 0.0
                    for i in range(N):
                        for (vx, vy) in verts_best[i]:
                            for (nx, ny) in normals_out:
                                v = (nx * (vx - ncx) + ny * (vy - ncy)) / apothem_unit
                                if v > val:
                                    val = v
                    if val < best_val:
                        best_val = val
                        best_cx, best_cy = ncx, ncy
            
            if abs(best_cx - cx_c) < 1e-5 and abs(best_cy - cy_c) < 1e-5:
                break
            cx_c, cy_c = best_cx, best_cy
        
        if max_proj < best_final_side:
            best_final_side = max_proj
            best_final_center = (cx_c, cy_c)
            best_final_angle = float(outer_angle)
    
    # Final verification
    cx_final = np.array([p[0] for p in best_final_cfg])
    cy_final = np.array([p[1] for p in best_final_cfg])
    angle_final = np.array([p[2] for p in best_final_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        cfg = make_lattice_initial()
        scale = 1.3
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_final_cfg = cfg
        cx_final = np.array([p[0] for p in cfg])
        cy_final = np.array([p[1] for p in cfg])
        angle_final = np.array([p[2] for p in cfg])
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
        best_final_center = (0.0, 0.0)
        best_final_angle = 0.0
    else:
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, best_final_center, best_final_angle)
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_final_cfg]
    return inner, best_final_center, best_final_side, best_final_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
