import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0

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

    def are_disjoint_sat(v1, v2, eps=1e-9):
        """
        Returns True if v1 and v2 are disjoint (including touching).
        """
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

    def compute_outer_side_vec(centers_arr, angles_arr, ocx, ocy, otheta):
        """Vectorized computation of outer side length."""
        max_proj = 0.0
        otheta_rad = math.radians(otheta)
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers_arr)):
                cx, cy = centers_arr[i]
                angle_rad = math.radians(angles_arr[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def objective(params):
        if time.time() > deadline - 8:
            return 1e9

        centers = []
        for i in range(12):
            centers.append([params[i * 2], params[i * 2 + 1]])
        angles = [params[24 + i] for i in range(12)]
        ocx, ocy = params[36], params[37]
        otheta = params[38]
        
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        
        # Soft penalty for overlaps based on center distance
        # Min distance for non-overlap is sqrt(3)
        min_dist = math.sqrt(3)
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return side + 10.0 * penalty

    def is_valid(centers, angles):
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
            
        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint_sat(verts[i], verts[j]):
                    return False
        return True

    def polish_outer_hexagon(centers, angles, ocx, ocy, otheta, max_iters=500):
        """High-precision polish of outer hexagon position and rotation."""
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
        
        # Multi-scale grid search
        for scale in [5.0, 1.0, 0.1, 0.01, 0.001]:
            for _ in range(max_iters // 5):
                improved = False
                # Try small perturbations in a local neighborhood
                for da in np.linspace(-scale, scale, 21):
                    for dx in np.linspace(-scale * 0.5, scale * 0.5, 11):
                        for dy in np.linspace(-scale * 0.5, scale * 0.5, 11):
                            s = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, best_otheta + da)
                            if s < best_side - 1e-10:
                                best_side = s
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta += da
                                improved = True
                if not improved:
                    break
        
        return best_ocx, best_ocy, best_otheta, best_side

    def optimize_from_start(x0):
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
            return res.x, res.fun
        except Exception:
            return x0, objective(x0)

    # Generate initial configurations
    starts = []
    
    # 1. Hexagonal Lattice
    d = math.sqrt(3) + 0.01
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
    
    centers_init = [np.array([0.0, 0.0])]
    ring1_vectors = [
        u, v, v - u, -u, -v, u - v
    ]
    for vec in ring1_vectors:
        centers_init.append(vec.copy())
        
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
        centers_init.append(ring2_candidates[i].copy())
        
    x0 = []
    for c in centers_init:
        x0.append(c[0])
        x0.append(c[1])
    for i in range(12):
        x0.append(30.0)
    x0.extend([0.0, 0.0, 0.0])
    starts.append(x0)
    
    # 2. Compressed Lattice (try to squeeze)
    d_comp = math.sqrt(3) * 0.95
    u_c = np.array([d_comp, 0.0])
    v_c = np.array([d_comp / 2.0, d_comp * math.sqrt(3) / 2.0])
    centers_comp = [np.array([0.0, 0.0])]
    ring1_c = [u_c, v_c, v_c - u_c, -u_c, -v_c, u_c - v_c]
    for vec in ring1_c:
        centers_comp.append(vec.copy())
    ring2_c = []
    for i in range(6):
        p1 = ring1_c[i]
        for b in [u_c, v_c, v_c - u_c, -u_c, -v_c, u_c - v_c]:
            p2 = p1 + b
            if not np.allclose(p2, np.array([0.0, 0.0])) and not any(np.allclose(p2, p1j) for p1j in ring1_c):
                if not any(np.allclose(p2, pc) for pc in ring2_c):
                    ring2_c.append(p2)
    ring2_c.sort(key=lambda p: np.linalg.norm(p))
    for i in range(5):
        centers_comp.append(ring2_c[i].copy())
        
    x1 = []
    for c in centers_comp:
        x1.append(c[0])
        x1.append(c[1])
    for i in range(12):
        x1.append(30.0)
    x1.extend([0.0, 0.0, 0.0])
    starts.append(x1)
    
    # 3. Random Starts
    random.seed(42)
    for _ in range(5):
        centers_rand = []
        for _ in range(12):
            centers_rand.append([random.uniform(-3, 3), random.uniform(-3, 3)])
        x_rand = []
        for c in centers_rand:
            x_rand.append(c[0])
            x_rand.append(c[1])
        for i in range(12):
            x_rand.append(random.uniform(0, 60))
        x_rand.extend([0.0, 0.0, 0.0])
        starts.append(x_rand)

    best_result = None
    best_val = float('inf')

    for x0 in starts:
        if time.time() > deadline - 10:
            break
        res_x, res_val = optimize_from_start(x0)
        if res_val < best_val:
            best_val = res_val
            best_result = res_x

    if best_result is None:
        best_result = starts[0]

    # Extract best parameters
    centers = []
    for i in range(12):
        centers.append([best_result[i * 2], best_result[i * 2 + 1]])
    angles = [best_result[24 + i] for i in range(12)]
    ocx, ocy = best_result[36], best_result[37]
    otheta = best_result[38]
    
    # Final Polish: High-precision grid search on outer parameters
    best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
    best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
    
    # Coarse-to-fine polish
    best_ocx, best_ocy, best_otheta, best_side = polish_outer_hexagon(
        centers, angles, best_ocx, best_ocy, best_otheta, max_iters=300
    )
    
    # Verify final validity
    if not is_valid(centers, angles):
        # Fallback to initial lattice if optimization resulted in invalid configuration
        centers = [[c[0], c[1]] for c in centers_init]
        angles = [30.0] * 12
        ocx, ocy, otheta = 0.0, 0.0, 0.0
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
        # Polish the fallback too
        best_ocx, best_ocy, best_otheta, best_side = polish_outer_hexagon(
            centers, angles, best_ocx, best_ocy, best_otheta, max_iters=200
        )

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (best_ocx, best_ocy)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
