import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s
INRADIUS = math.sqrt(3) / 2.0
MIN_DIST_CIRCLE = 2 * INRADIUS  # 2 * sqrt(3)/2 = sqrt(3) ~ 1.732

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

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    
    # Fast pre-check: Circle packing
    # If distance between centers < 2*inradius, they MUST overlap.
    # This is a necessary condition, not sufficient.
    for i in range(n):
        xi, yi = cx_arr[i], cy_arr[i]
        for j in range(i + 1, n):
            xj, yj = cx_arr[j], cy_arr[j]
            dx = xi - xj
            dy = yi - yj
            if dx*dx + dy*dy < (MIN_DIST_CIRCLE * MIN_DIST_CIRCLE) - 1e-9:
                return False

    # Exact check: SAT
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            vj = verts_list[j]
            aj = axes_list[j]
            
            # We already know dist > 2*inradius, but corners might still touch.
            # SAT check
            for axis in ai:
                minA, maxA = _project_polygon(vi, axis)
                minB, maxB = _project_polygon(vj, axis)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    break
            else:
                for axis in aj:
                    minA, maxA = _project_polygon(vi, axis)
                    minB, maxB = _project_polygon(vj, axis)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        break
                else:
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

def _quick_score(cx_arr, cy_arr, angle_arr):
    # Approximate score assuming outer hex is centered at (0,0) and aligned 0
    apothem_unit = math.sqrt(3) / 2.0
    normals = [(math.cos(math.radians(90 + 60 * k)), math.sin(math.radians(90 + 60 * k))) for k in range(6)]
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

    def make_ring_initial():
        pts = [[0.0, 0.0, 0.0]]
        r = 2 * math.sqrt(3)
        for i in range(11):
            angle = math.radians(360 * i / 11 + 15)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            pts.append([x, y, 0.0])
        return pts

    def make_random_initial():
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    initializers = [
        make_flower_initial,
        make_lattice_initial,
        make_ring_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    
    # 1. Initialize
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 100:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side = _quick_score(cx_arr, cy_arr, angle_arr)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
    
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
        best_side = _quick_score(cx_arr, cy_arr, angle_arr)

    # 2. Differential Evolution (Global Search)
    cx_arr = np.array([p[0] for p in best_cfg])
    cy_arr = np.array([p[1] for p in best_cfg])
    angle_arr = np.array([p[2] for p in best_cfg])
    
    pop_size = 30
    population = []
    population.append([cx_arr.copy(), cy_arr.copy(), angle_arr.copy(), best_side])
    
    for _ in range(pop_size - 1):
        p = best_cfg[:]
        for i in range(N):
            p[i][0] += random.gauss(0, 0.5)
            p[i][1] += random.gauss(0, 0.5)
            p[i][2] += random.gauss(0, 10.0)
        
        pcx = np.array([q[0] for q in p])
        pcy = np.array([q[1] for q in p])
        pang = np.array([q[2] for q in p])
        
        if _is_valid_fast(pcx, pcy, pang):
            score = _quick_score(pcx, pcy, pang)
            population.append([pcx, pcy, pang, score])
        else:
            population.append([pcx, pcy, pang, float('inf')])
    
    F = 0.8
    CR = 0.9
    de_generations = 100 # Reduced to leave time for SA/Polish
    
    for gen in range(de_generations):
        if time.time() - t0 > TIME_LIMIT - 50:
            break
        
        for i in range(pop_size):
            if time.time() - t0 > TIME_LIMIT - 50:
                break
            
            a, b, c = random.sample(range(pop_size), 3)
            
            trial_cx = population[a][0].copy()
            trial_cy = population[a][1].copy()
            trial_ang = population[a][2].copy()
            
            for j in range(N):
                trial_cx[j] = population[a][0][j] + F * (population[b][0][j] - population[c][0][j])
                trial_cy[j] = population[a][1][j] + F * (population[b][1][j] - population[c][1][j])
                trial_ang[j] = population[a][2][j] + F * (population[b][2][j] - population[c][2][j])
            
            j_rand = random.randrange(N)
            for j in range(N):
                if random.random() < CR or j == j_rand:
                    cx_arr[j] = trial_cx[j]
                    cy_arr[j] = trial_cy[j]
                    angle_arr[j] = trial_ang[j]
            
            if _is_valid_fast(cx_arr, cy_arr, angle_arr):
                new_score = _quick_score(cx_arr, cy_arr, angle_arr)
                if new_score < population[i][3]:
                    population[i][0] = cx_arr.copy()
                    population[i][1] = cy_arr.copy()
                    population[i][2] = angle_arr.copy()
                    population[i][3] = new_score
                    
                    if new_score < best_side:
                        best_side = new_score
                        best_cfg = [[cx_arr[k], cy_arr[k], angle_arr[k]] for k in range(N)]
                else:
                    cx_arr[:] = population[i][0]
                    cy_arr[:] = population[i][1]
                    angle_arr[:] = population[i][2]
            else:
                cx_arr[:] = population[i][0]
                cy_arr[:] = population[i][1]
                angle_arr[:] = population[i][2]

    # 3. Simulated Annealing (Local Refinement with Pairwise Moves)
    cx_arr = np.array([p[0] for p in best_cfg])
    cy_arr = np.array([p[1] for p in best_cfg])
    angle_arr = np.array([p[2] for p in best_cfg])
    
    cur_score = best_side
    T = 0.2
    T_end = 0.001
    alpha = 0.998
    
    while time.time() - t0 < TIME_LIMIT - 20:
        idx = random.randrange(N)
        old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
        
        # 80% Single Move, 20% Pairwise Move
        if random.random() < 0.8:
            cx_arr[idx] += random.gauss(0, 0.1)
            cy_arr[idx] += random.gauss(0, 0.1)
            angle_arr[idx] += random.gauss(0, 2.0)
        else:
            idx2 = random.randrange(N)
            if idx2 == idx: continue
            old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
            
            cx_arr[idx] += random.gauss(0, 0.1)
            cy_arr[idx] += random.gauss(0, 0.1)
            angle_arr[idx] += random.gauss(0, 2.0)
            
            cx_arr[idx2] += random.gauss(0, 0.1)
            cy_arr[idx2] += random.gauss(0, 0.1)
            angle_arr[idx2] += random.gauss(0, 2.0)
            
            valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
            
            if valid:
                new_score = _quick_score(cx_arr, cy_arr, angle_arr)
                if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                    cur_score = new_score
                    if new_score < best_side:
                        best_side = new_score
                        best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
                else:
                    cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                    cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
            T = max(T_end, T * alpha)
            continue
        
        valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
        
        if valid:
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

    # 4. Final Polish: Optimize Outer Hexagon
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = (0.0, 0.0)
    best_final_angle = 0.0
    
    verts_best = [_hex_vertices(cx_best[i], cy_best[i], angle_best[i]) for i in range(N)]
    
    # Coarse search for outer angle and center
    for outer_angle in range(0, 30, 5):
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        
        a = math.radians(outer_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals_out = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals_out.append((math.cos(na), math.sin(na)))
        
        cx_c, cy_c = 0.0, 0.0
        
        # Local search for center
        for _iter in range(10):
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_best[i]:
                    for (nx, ny) in normals_out:
                        val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                        if val > max_proj:
                            max_proj = val
            
            best_cx, best_cy, best_val = cx_c, cy_c, max_proj
            
            # Check neighbors
            for dx in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                for dy in [-0.2, -0.1, 0.0, 0.1, 0.2]:
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
            
            if abs(best_cx - cx_c) < 1e-4 and abs(best_cy - cy_c) < 1e-4:
                break
            cx_c, cy_c = best_cx, best_cy
        
        if max_proj < best_final_side:
            best_final_side = max_proj
            best_final_center = (cx_c, cy_c)
            best_final_angle = float(outer_angle)
    
    # Final verification and exact side calculation
    cx_final = np.array([p[0] for p in best_final_cfg])
    cy_final = np.array([p[1] for p in best_final_cfg])
    angle_final = np.array([p[2] for p in best_final_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        # Fallback
        cfg = make_lattice_initial()
        scale = 1.5
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
