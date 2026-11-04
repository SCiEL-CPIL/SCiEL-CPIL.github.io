import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        max_proj = 0.0
        otheta_rad = math.radians(otheta_deg)
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def is_valid_sat(centers, angles, eps=1e-9):
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        
        def project(vertices, nx, ny):
            projs = np.dot(vertices, [nx, ny])
            return np.min(projs), np.max(projs)

        def are_disjoint(v1, v2):
            for i in range(6):
                p1 = v1[i]
                p2 = v1[(i + 1) % 6]
                edge = p2 - p1
                nx = -edge[1]
                ny = edge[0]
                min1, max1 = project(v1, nx, ny)
                min2, max2 = project(v2, nx, ny)
                if max1 < min2 + eps or max2 < min1 + eps:
                    return True
            for i in range(6):
                p1 = v2[i]
                p2 = v2[(i + 1) % 6]
                edge = p2 - p1
                nx = -edge[1]
                ny = edge[0]
                min1, max1 = project(v1, nx, ny)
                min2, max2 = project(v2, nx, ny)
                if max1 < min2 + eps or max2 < min1 + eps:
                    return True
            return False

        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint(verts[i], verts[j]):
                    return False
        return True

    def polish_outer(centers, angles, initial_ocx, initial_ocy, initial_otheta):
        best_side = compute_outer_side(centers, angles, initial_ocx, initial_ocy, initial_otheta)
        best_ocx, best_ocy, best_otheta = initial_ocx, initial_ocy, initial_otheta
        
        # Nelder-Mead refinement
        def obj_outer(x):
            return compute_outer_side(centers, angles, x[0], x[1], x[2])
        
        try:
            res = minimize(obj_outer, [best_ocx, best_ocy, best_otheta], method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 200})
            if res.success:
                best_ocx, best_ocy, best_otheta = res.x[0], res.x[1], res.x[2]
                best_side = compute_outer_side(centers, angles, best_ocx, best_ocy, best_otheta)
        except Exception:
            pass
        
        # Multi-scale grid polish
        scales = [0.01, 0.001, 0.0001, 0.00001]
        for scale in scales:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da_deg in [-scale*10, 0, scale*10]:
                            new_otheta = best_otheta + da_deg
                            side = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, new_otheta)
                            if side < best_side - 1e-12:
                                best_side = side
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta = new_otheta
                                improved = True
        return best_ocx, best_ocy, best_otheta, best_side

    def optimize_slsqp(x0, max_iters=80):
        n = 12
        def objective(x):
            if time.time() > deadline - 25:
                return 1e10
            centers = [[x[i*3], x[i*3+1]] for i in range(n)]
            angles = [x[i*3+2] for i in range(n)]
            ocx, ocy, otheta = x[3*n], x[3*n+1], x[3*n+2]
            return compute_outer_side(centers, angles, ocx, ocy, otheta)
        
        def constraint_fun(x, i, j):
            xi = x[i*3]
            yi = x[i*3+1]
            xj = x[j*3]
            yj = x[j*3+1]
            return (xi - xj)**2 + (yi - yj)**2 - 3.0
        
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: constraint_fun(x, i, j)})
        
        bounds = []
        for i in range(n):
            bounds.extend([(-5, 5), (-5, 5), (0, 60)])
        bounds.extend([(-2, 2), (-2, 2), (0, 60)])
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': max_iters, 'ftol': 1e-10})
            if res.success:
                return res.x
        except Exception:
            pass
        return x0

    # Generate initial configurations
    eps = 0.001
    d = math.sqrt(3) + eps
    
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    ring1_vectors = [u, v, v-u, -u, -v, u-v]
    
    # Generate ring 2 candidates
    ring2_candidates = []
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in [u, v, v-u, -u, -v, u-v]:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]), atol=0.01)
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j, atol=0.01):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc, atol=0.01):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
    
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    
    # Build multiple starting configurations
    starts = []
    
    # Config 0: closest 5 ring-2 points
    ring2_sel_0 = ring2_candidates[:5]
    centers0 = [[0.0, 0.0]] + [[vec[0], vec[1]] for vec in ring1_vectors] + [[vec[0], vec[1]] for vec in ring2_sel_0]
    x0 = []
    for c in centers0:
        x0.extend([c[0], c[1], 30.0])
    x0.extend([0.0, 0.0, 0.0])
    starts.append(x0)
    
    # Config 1-3: different ring-2 selections
    random.seed(42)
    for cfg_idx in range(3):
        idxs = list(range(len(ring2_candidates)))
        random.shuffle(idxs)
        ring2_sel = [ring2_candidates[i] for i in idxs[:5]]
        centers_cfg = [[0.0, 0.0]] + [[vec[0], vec[1]] for vec in ring1_vectors] + [[vec[0], vec[1]] for vec in ring2_sel]
        x_cfg = []
        for c in centers_cfg:
            x_cfg.extend([c[0], c[1], 30.0])
        x_cfg.extend([0.0, 0.0, 0.0])
        starts.append(x_cfg)
    
    # Config 4: random start
    random.seed(123)
    centers_rand = []
    for _ in range(12):
        centers_rand.append([random.uniform(-2, 2), random.uniform(-2, 2)])
    x_rand = []
    for c in centers_rand:
        x_rand.extend([c[0], c[1], random.uniform(0, 60)])
    x_rand.extend([0.0, 0.0, 0.0])
    starts.append(x_rand)
    
    best_result = None
    best_val = float('inf')

    for x0 in starts:
        if time.time() > deadline - 30:
            break
        
        res_x = optimize_slsqp(x0)
        
        centers = [[res_x[i*3], res_x[i*3+1]] for i in range(12)]
        angles = [res_x[i*3+2] for i in range(12)]
        ocx, ocy = res_x[36], res_x[37]
        otheta = res_x[38]
        
        best_ocx, best_ocy, best_otheta, best_side = polish_outer(
            centers, angles, ocx, ocy, otheta
        )
        
        if is_valid_sat(centers, angles):
            if best_side < best_val:
                best_val = best_side
                best_result = (centers, angles, best_ocx, best_ocy, best_otheta, best_side)

    if best_result is None:
        centers = [[c[0], c[1]] for c in centers0]
        angles = [30.0] * 12
        ocx, ocy, otheta = 0.0, 0.0, 0.0
        best_ocx, best_ocy, best_otheta, best_side = polish_outer(
            centers, angles, ocx, ocy, otheta
        )
        best_result = (centers, angles, best_ocx, best_ocy, best_otheta, best_side)

    centers, angles, ocx, ocy, otheta, side = best_result
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (ocx, ocy)
    outer_side = side
    outer_angle = otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
