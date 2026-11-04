import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0
    random.seed(42)

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        thetas = angle + np.arange(6) * (math.pi / 3)
        xs = cx + side * np.cos(thetas)
        ys = cy + side * np.sin(thetas)
        return np.column_stack((xs, ys))

    def are_disjoint_sat(v1, v2, eps=1e-9):
        e1 = np.roll(v1, -1, axis=0) - v1
        n1 = np.column_stack((-e1[:, 1], e1[:, 0]))
        
        p1 = v1 @ n1.T
        p2 = v2 @ n1.T
        
        min1 = np.min(p1, axis=0)
        max1 = np.max(p1, axis=0)
        min2 = np.min(p2, axis=0)
        max2 = np.max(p2, axis=0)
        
        if np.any(max1 < min2 + eps) or np.any(max2 < min1 + eps):
            return True

        e2 = np.roll(v2, -1, axis=0) - v2
        n2 = np.column_stack((-e2[:, 1], e2[:, 0]))
        
        p1 = v1 @ n2.T
        p2 = v2 @ n2.T
        
        min1 = np.min(p1, axis=0)
        max1 = np.max(p1, axis=0)
        min2 = np.min(p2, axis=0)
        max2 = np.max(p2, axis=0)
        
        if np.any(max1 < min2 + eps) or np.any(max2 < min1 + eps):
            return True

        return False

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        max_proj = 0.0
        otheta_rad = math.radians(otheta)
        normal_angles = otheta_rad + math.pi / 6 + np.arange(6) * (math.pi / 3)
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        for i in range(len(centers)):
            cx, cy = centers[i]
            angle_rad = math.radians(angles[i])
            thetas = angle_rad + np.arange(6) * (math.pi / 3)
            vx = cx + np.cos(thetas)
            vy = cy + np.sin(thetas)
            
            dx = vx - ocx
            dy = vy - ocy
            projs = dx[:, np.newaxis] * nx[np.newaxis, :] + dy[:, np.newaxis] * ny[np.newaxis, :]
            current_max = np.max(projs)
            if current_max > max_proj:
                max_proj = current_max
        
        return 2.0 * max_proj / math.sqrt(3)

    def calculate_cost(centers, angles, ocx, ocy, otheta):
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        
        penalty = 0.0
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
            
        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint_sat(verts[i], verts[j]):
                    penalty += 100.0
        
        return side + penalty

    def optimize_sa(centers_init, angles_init, ocx_init, ocy_init, otheta_init):
        T = 0.3
        T_min = 0.0005
        alpha = 0.985
        steps_per_temp = 15
        
        current_centers = [c.copy() for c in centers_init]
        current_angles = list(angles_init)
        current_ocx = ocx_init
        current_ocy = ocy_init
        current_otheta = otheta_init
        
        current_cost = calculate_cost(current_centers, current_angles, current_ocx, current_ocy, current_otheta)
        
        best_centers = [c.copy() for c in current_centers]
        best_angles = list(current_angles)
        best_ocx = current_ocx
        best_ocy = current_ocy
        best_otheta = current_otheta
        best_cost = current_cost
        
        while T > T_min:
            for _ in range(steps_per_temp):
                if time.time() > deadline - 15:
                    break
                
                new_centers = [c.copy() for c in current_centers]
                new_angles = list(current_angles)
                new_ocx = current_ocx
                new_ocy = current_ocy
                new_otheta = current_otheta
                
                i = random.randint(0, 11)
                step_size = T * 0.15
                new_centers[i][0] += random.gauss(0, step_size)
                new_centers[i][1] += random.gauss(0, step_size)
                new_angles[i] += random.gauss(0, step_size * 3)
                
                if random.random() < 0.08:
                    new_ocx += random.gauss(0, T * 0.05)
                    new_ocy += random.gauss(0, T * 0.05)
                    new_otheta += random.gauss(0, T * 0.05)
                
                new_cost = calculate_cost(new_centers, new_angles, new_ocx, new_ocy, new_otheta)
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_centers = new_centers
                    current_angles = new_angles
                    current_ocx = new_ocx
                    current_ocy = new_ocy
                    current_otheta = new_otheta
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_centers = [c.copy() for c in current_centers]
                        best_angles = list(current_angles)
                        best_ocx = current_ocx
                        best_ocy = current_ocy
                        best_otheta = current_otheta
            
            T *= alpha
            if time.time() > deadline - 15:
                break
                
        return best_centers, best_angles, best_ocx, best_ocy, best_otheta, best_cost

    def generate_lattice_config():
        d = math.sqrt(3) + 0.005
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
            
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
            centers.append(ring2_candidates[i].copy())
            
        angles = [30.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    def generate_circular_config():
        centers = [np.array([0.0, 0.0])]
        r = math.sqrt(3) + 0.01
        for i in range(6):
            angle = i * math.pi / 3
            cx = r * math.cos(angle)
            cy = r * math.sin(angle)
            centers.append(np.array([cx, cy]))
        
        r2 = 2 * math.sqrt(3) + 0.01
        for i in range(5):
            angle = i * 2 * math.pi / 5 + math.pi / 5
            cx = r2 * math.cos(angle)
            cy = r2 * math.sin(angle)
            centers.append(np.array([cx, cy]))
            
        angles = [30.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    def generate_row_config():
        centers = []
        d = math.sqrt(3) + 0.005
        for i in range(6):
            centers.append(np.array([i * d, 0.0]))
        for i in range(6):
            centers.append(np.array([i * d + d/2, d * math.sqrt(3)/2]))
            
        angles = [30.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    def generate_staggered_config():
        centers = []
        d = math.sqrt(3) + 0.005
        h = d * math.sqrt(3) / 2
        for row in range(3):
            y = row * h
            x_offset = (row % 2) * d / 2
            for col in range(4):
                centers.append(np.array([col * d + x_offset, y]))
        angles = [30.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    configs = [
        generate_lattice_config(),
        generate_circular_config(),
        generate_row_config(),
        generate_staggered_config()
    ]
    
    best_overall_cost = float('inf')
    best_centers = None
    best_angles = None
    best_ocx = 0.0
    best_ocy = 0.0
    best_otheta = 0.0
    
    for centers_init, angles_init, ocx_init, ocy_init, otheta_init in configs:
        if time.time() > deadline - 25:
            break
        
        sa_centers, sa_angles, sa_ocx, sa_ocy, sa_otheta, sa_cost = optimize_sa(
            centers_init, angles_init, ocx_init, ocy_init, otheta_init
        )
        
        def outer_objective(v):
            if time.time() > deadline - 5:
                return 1e6
            ox, oy, oa = v[0], v[1], v[2]
            return compute_outer_side(sa_centers, sa_angles, ox, oy, oa)
        
        result = minimize(outer_objective, [sa_ocx, sa_ocy, sa_otheta], method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        pol_ocx, pol_ocy, pol_otheta = result.x
        pol_cost = result.fun
        
        verts = []
        for i in range(12):
            v = get_hex_vertices(sa_centers[i][0], sa_centers[i][1], 1.0, sa_angles[i])
            verts.append(v)
            
        valid = True
        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint_sat(verts[i], verts[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid and pol_cost < best_overall_cost:
            best_overall_cost = pol_cost
            best_centers = sa_centers
            best_angles = sa_angles
            best_ocx = pol_ocx
            best_ocy = pol_ocy
            best_otheta = pol_otheta
    
    if best_centers is None:
        best_centers, best_angles, best_ocx, best_ocy, best_otheta = generate_lattice_config()
        best_overall_cost = compute_outer_side(best_centers, best_angles, best_ocx, best_ocy, best_otheta)
    
    best_side = best_overall_cost
    final_ocx, final_ocy, final_otheta = best_ocx, best_ocy, best_otheta
    
    # Multi-stage polish
    for stage in range(3):
        ranges = [(20, 11), (5, 11), (1, 11)]
        dx_ranges = [(0.5, 7), (0.1, 7), (0.02, 7)]
        dy_ranges = [(0.5, 7), (0.1, 7), (0.02, 7)]
        
        da_range, da_n = ranges[stage]
        dx_range, dx_n = dx_ranges[stage]
        dy_range, dy_n = dy_ranges[stage]
        
        for da in np.linspace(-da_range, da_range, da_n):
            for dx in np.linspace(-dx_range, dx_range, dx_n):
                for dy in np.linspace(-dy_range, dy_range, dy_n):
                    if time.time() > deadline - 2:
                        break
                    s = compute_outer_side(best_centers, best_angles, final_ocx + dx, final_ocy + dy, final_otheta + da)
                    if s < best_side:
                        best_side = s
                        final_ocx, final_ocy, final_otheta = final_ocx + dx, final_ocy + dy, final_otheta + da
                if time.time() > deadline - 2:
                    break
            if time.time() > deadline - 2:
                break
        if time.time() > deadline - 2:
            break
    
    if time.time() < deadline - 1:
        for da in np.linspace(-0.5, 0.5, 21):
            for dx in np.linspace(-0.01, 0.01, 11):
                for dy in np.linspace(-0.01, 0.01, 11):
                    s = compute_outer_side(best_centers, best_angles, final_ocx + dx, final_ocy + dy, final_otheta + da)
                    if s < best_side:
                        best_side = s
                        final_ocx, final_ocy, final_otheta = final_ocx + dx, final_ocy + dy, final_otheta + da

    inner = [[float(c[0]), float(c[1]), float(a)] for c, a in zip(best_centers, best_angles)]
    outer_center = (float(final_ocx), float(final_ocy))
    outer_side = float(best_side)
    outer_angle = float(final_otheta)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
