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
        """Vectorized computation of outer side length."""
        otheta_rad = math.radians(otheta)
        # Precompute all 72 vertex positions (12 hexagons x 6 vertices)
        n = len(centers)
        cx_arr = np.array([c[0] for c in centers])
        cy_arr = np.array([c[1] for c in centers])
        ang_rad = np.radians(np.array(angles))
        
        # Vertex offsets: 6 vertices per hexagon
        vert_angles = ang_rad[:, None] + np.arange(6)[None, :] * math.pi / 3  # (n, 6)
        vx = cx_arr[:, None] + np.cos(vert_angles)  # (n, 6)
        vy = cy_arr[:, None] + np.sin(vert_angles)  # (n, 6)
        
        # Flatten to (72,)
        vx_flat = vx.flatten()
        vy_flat = vy.flatten()
        
        # 6 normal directions
        normal_angles = otheta_rad + math.pi / 6 + np.arange(6) * math.pi / 3
        nx_arr = np.cos(normal_angles)  # (6,)
        ny_arr = np.sin(normal_angles)  # (6,)
        
        # Project all vertices onto all 6 normals: (72, 6)
        projs = (vx_flat - ocx)[:, None] * nx_arr[None, :] + (vy_flat - ocy)[:, None] * ny_arr[None, :]
        max_proj = np.max(projs)
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

    def objective_vec(params):
        """Faster objective using vectorized compute_outer_side."""
        if time.time() > deadline - 8:
            return 1e9

        centers = []
        for i in range(12):
            centers.append([params[i * 2], params[i * 2 + 1]])
        angles = [params[24 + i] for i in range(12)]
        ocx, ocy = params[36], params[37]
        otheta = params[38]
        
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        
        # Vectorized penalty
        min_dist = math.sqrt(3)
        cx_arr = np.array([c[0] for c in centers])
        cy_arr = np.array([c[1] for c in centers])
        penalty = 0.0
        for i in range(12):
            dx = cx_arr[i+1:] - cx_arr[i]
            dy = cy_arr[i+1:] - cy_arr[i]
            dists = np.sqrt(dx*dx + dy*dy)
            violations = np.maximum(0, min_dist - dists)
            penalty += np.sum(violations ** 2)
        
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
        """High-precision polish of outer hexagon position and rotation using Nelder-Mead."""
        def outer_obj(params):
            return compute_outer_side(centers, angles, params[0], params[1], params[2])
        
        x0 = np.array([ocx, ocy, otheta])
        try:
            res = minimize(outer_obj, x0, method='Nelder-Mead',
                           options={'maxiter': max_iters, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < compute_outer_side(centers, angles, ocx, ocy, otheta):
                return res.x[0], res.x[1], res.x[2], res.fun
        except Exception:
            pass
        
        # Fallback: fine grid search
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
        for scale in [0.1, 0.01, 0.001, 0.0001]:
            improved = False
            for da in np.linspace(-scale, scale, 11):
                for dx in np.linspace(-scale, scale, 11):
                    for dy in np.linspace(-scale, scale, 11):
                        s = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, best_otheta + da)
                        if s < best_side - 1e-12:
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
            res = minimize(objective_vec, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
            return res.x, res.fun
        except Exception:
            return x0, objective_vec(x0)

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
    
    # Final Polish: High-precision optimization on outer parameters
    best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
    best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
    
    # Coarse-to-fine polish
    best_ocx, best_ocy, best_otheta, best_side = polish_outer_hexagon(
        centers, angles, best_ocx, best_ocy, best_otheta, max_iters=500
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
            centers, angles, best_ocx, best_ocy, best_otheta, max_iters=300
        )

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (best_ocx, best_ocy)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
