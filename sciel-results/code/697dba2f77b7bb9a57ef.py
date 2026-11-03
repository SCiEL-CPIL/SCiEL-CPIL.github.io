import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    random.seed(42)

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def project(vertices, nx, ny):
        projs = np.dot(vertices, [nx, ny])
        return np.min(projs), np.max(projs)

    def get_separation_axes(v1, v2):
        axes = []
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            axes.append((-edge[1], edge[0]))
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            axes.append((-edge[1], edge[0]))
        return axes

    def min_distance_hex(v1, v2):
        min_sep = float('inf')
        overlap_depths = []
        
        axes = get_separation_axes(v1, v2)
        for nx, ny in axes:
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            
            if max1 < min2 or max2 < min1:
                sep = max(min2 - max1, min1 - max2)
                min_sep = min(min_sep, sep)
            else:
                overlap = min(max1, max2) - max(min1, min2)
                overlap_depths.append(overlap)
                
        if min_sep < float('inf'):
            return min_sep
        else:
            if overlap_depths:
                return -min(overlap_depths)
            else:
                return 0.0

    def calc_distance_penalty(centers, angles):
        verts = []
        for i in range(N):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = min_distance_hex(verts[i], verts[j])
                if dist < 0:
                    penalty += (dist * dist) * 100.0
        return penalty

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        max_proj = 0.0
        otheta_rad = math.radians(otheta)
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

    def objective(params):
        centers = np.array(params[0:24]).reshape(12, 2)
        angles = params[24:36]
        ocx, ocy, otheta = params[36], params[37], params[38]
        
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        penalty = calc_distance_penalty(centers, angles)
        
        return side + penalty

    def optimize_start(centers_init, angles_init, ocx_init, ocy_init, otheta_init):
        x0 = np.zeros(47)
        for i in range(12):
            x0[2*i] = centers_init[i][0]
            x0[2*i+1] = centers_init[i][1]
            x0[24+i] = angles_init[i]
        x0[36] = ocx_init
        x0[37] = ocy_init
        x0[38] = otheta_init
        
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        return res.x, res.fun

    d = math.sqrt(3) + 0.01
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
    
    centers_init1 = [np.array([0.0, 0.0])]
    ring1_vectors = [u, v, v - u, -u, -v, u - v]
    for vec in ring1_vectors:
        centers_init1.append(vec.copy())
        
    ring2_candidates = []
    basis = [u, v, v - u, -u, -v, u - v]
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
                    
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    for i in range(5):
        centers_init1.append(ring2_candidates[i].copy())
        
    angles_init1 = [30.0] * 12
    ocx_init1 = 0.0
    ocy_init1 = 0.0
    otheta_init1 = 0.0

    centers_init2 = [c + np.random.normal(0, 0.1, 2) for c in centers_init1]
    angles_init2 = [a + np.random.normal(0, 5) for a in angles_init1]
    ocx_init2 = 0.0
    ocy_init2 = 0.0
    otheta_init2 = 0.0

    best_params = None
    best_cost = float('inf')
    
    starts = [
        (centers_init1, angles_init1, ocx_init1, ocy_init1, otheta_init1),
        (centers_init2, angles_init2, ocx_init2, ocy_init2, otheta_init2)
    ]
    
    for start in starts:
        if time.time() > deadline - 20:
            break
        p, c = optimize_start(*start)
        if c < best_cost:
            best_cost = c
            best_params = p
            
    if best_params is None:
        best_params = np.zeros(47)
        for i in range(12):
            best_params[2*i] = centers_init1[i][0]
            best_params[2*i+1] = centers_init1[i][1]
            best_params[24+i] = angles_init1[i]
        best_params[36] = 0
        best_params[37] = 0
        best_params[38] = 0
        best_cost = objective(best_params)

    centers = np.array(best_params[0:24]).reshape(12, 2)
    angles = best_params[24:36]
    ocx, ocy, otheta = best_params[36], best_params[37], best_params[38]
    
    best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
    final_ocx, final_ocy, final_otheta = ocx, ocy, otheta
    
    for da in np.linspace(-5, 5, 11):
        for dx in np.linspace(-0.2, 0.2, 5):
            for dy in np.linspace(-0.2, 0.2, 5):
                s = compute_outer_side(centers, angles, ocx + dx, ocy + dy, otheta + da)
                if s < best_side:
                    best_side = s
                    final_ocx, final_ocy, final_otheta = ocx + dx, ocy + dy, otheta + da
                    
    verts = []
    for i in range(12):
        v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        verts.append(v)
        
    valid = True
    for i in range(12):
        for j in range(i + 1, 12):
            dist = min_distance_hex(verts[i], verts[j])
            if dist < 0:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        centers = np.array(centers_init1)
        angles = np.array(angles_init1)
        final_ocx, final_ocy, final_otheta = 0.0, 0.0, 0.0
        best_side = compute_outer_side(centers, angles, final_ocx, final_ocy, final_otheta)

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (final_ocx, final_ocy)
    outer_side = best_side
    outer_angle = final_otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
