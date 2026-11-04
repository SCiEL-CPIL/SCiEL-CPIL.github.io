import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0
    min_time_margin = 8.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def project_range(vertices, nx, ny):
        projs = np.dot(vertices, [nx, ny])
        return np.min(projs), np.max(projs)

    def compute_outer_side(centers, angles, otheta):
        """Compute the outer hexagon side length needed to contain all inner hexagons.
        Outer center is fixed at (0,0). otheta in radians."""
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def min_sat_separation(centers, angles, i, j):
        """Compute the minimum SAT separation between hexagons i and j.
        Positive means separated, negative means overlapping."""
        vi = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        vj = get_hex_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
        min_sep = float('inf')
        for k in range(6):
            p1 = vi[k]
            p2 = vi[(k + 1) % 6]
            edge = p2 - p1
            nx, ny = -edge[1], edge[0]
            min1, max1 = project_range(vi, nx, ny)
            min2, max2 = project_range(vj, nx, ny)
            sep = min(max1 - min2, max2 - min1)
            if sep < min_sep:
                min_sep = sep
        for k in range(6):
            p1 = vj[k]
            p2 = vj[(k + 1) % 6]
            edge = p2 - p1
            nx, ny = -edge[1], edge[0]
            min1, max1 = project_range(vi, nx, ny)
            min2, max2 = project_range(vj, nx, ny)
            sep = min(max1 - min2, max2 - min1)
            if sep < min_sep:
                min_sep = sep
        return min_sep

    def is_valid(centers, angles, eps=1e-7):
        for i in range(12):
            for j in range(i + 1, 12):
                if min_sat_separation(centers, angles, i, j) < -eps:
                    return False
        return True

    def total_overlap_penalty(centers, angles, alpha=30.0):
        """Smooth soft penalty for overlaps."""
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                sep = min_sat_separation(centers, angles, i, j)
                if sep < 0:
                    penalty += math.exp(alpha * sep) / alpha
        return penalty

    def generate_hexagonal_cluster():
        d = math.sqrt(3) + 0.01
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
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
        angles = [30.0] * 12
        return centers, angles

    def generate_rect3x4():
        d = math.sqrt(3) + 0.01
        dx = d
        dy = d * math.sqrt(3) / 2.0
        centers = []
        angles = []
        for row in range(4):
            for col in range(3):
                cx = (col - 1) * dx
                cy = (row - 1.5) * dy
                centers.append(np.array([cx, cy]))
                angles.append(30.0)
        return centers, angles

    def generate_hex6_plus6():
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        offset = 2.0
        centers = []
        angles = []
        for vec in [np.array([0.0, 0.0])] + ring1_vectors:
            centers.append(vec - np.array([offset, 0.0]))
            angles.append(30.0)
        for vec in [np.array([0.0, 0.0])] + ring1_vectors:
            centers.append(vec + np.array([offset, 0.0]))
            angles.append(30.0)
        return centers, angles

    def params_to_state(params):
        """Convert 37-dim param vector to (centers, angles, otheta_deg)."""
        centers = [[params[i * 2], params[i * 2 + 1]] for i in range(12)]
        angles = [params[24 + i] for i in range(12)]
        otheta = params[36]
        return centers, angles, otheta

    def state_to_params(centers, angles, otheta):
        """Convert state to 37-dim param vector."""
        params = []
        for cx, cy in centers:
            params.append(cx)
            params.append(cy)
        for a in angles:
            params.append(a)
        params.append(otheta)
        return params

    def slsqp_optimize(initial_params, max_iter=200):
        """SLSQP with SAT-based non-overlap inequality constraints."""
        def objective(params):
            if time.time() > deadline - min_time_margin:
                return 1e9
            centers, angles, otheta = params_to_state(params)
            return compute_outer_side(centers, angles, math.radians(otheta))

        def make_constraints():
            cons = []
            for i in range(12):
                for j in range(i + 1, 12):
                    def sep_constraint(params, i=i, j=j):
                        if time.time() > deadline - min_time_margin:
                            return 1.0
                        centers, angles, _ = params_to_state(params)
                        return min_sat_separation(centers, angles, i, j) + 1e-6
                    cons.append({'type': 'ineq', 'fun': sep_constraint})
            return cons

        try:
            res = minimize(objective, initial_params, method='SLSQP',
                           constraints=make_constraints(),
                           options={'maxiter': max_iter, 'ftol': 1e-9, 'disp': False})
            return res.x, res.fun
        except Exception:
            return initial_params, objective(initial_params)

    def nelder_polish(params, max_iter=500):
        """Nelder-Mead with soft overlap penalty in objective."""
        def obj(params):
            if time.time() > deadline - min_time_margin:
                return 1e9
            centers, angles, otheta = params_to_state(params)
            side = compute_outer_side(centers, angles, math.radians(otheta))
            penalty = total_overlap_penalty(centers, angles, alpha=30.0)
            return side + penalty

        try:
            res = minimize(obj, params, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-9})
            return res.x, res.fun
        except Exception:
            return params, obj(params)

    def polish_outer_angle(centers, angles, otheta_deg):
        """Multi-scale grid search on outer angle."""
        best_side = compute_outer_side(centers, angles, math.radians(otheta_deg))
        best_otheta = otheta_deg
        for scale in [5.0, 1.0, 0.1, 0.01]:
            for da in np.linspace(-scale, scale, int(4 * scale * 10) + 1):
                s = compute_outer_side(centers, angles, math.radians(otheta_deg + da))
                if s < best_side:
                    best_side = s
                    best_otheta = otheta_deg + da
        return best_otheta, best_side

    def center_config(centers):
        """Re-center configuration so centroid is at origin."""
        cx_mean = sum(c[0] for c in centers) / len(centers)
        cy_mean = sum(c[1] for c in centers) / len(centers)
        return [[c[0] - cx_mean, c[1] - cy_mean] for c in centers]

    # Generate initial configurations
    random.seed(42)
    configs = []

    # Config 1: Standard hex lattice
    c1, a1 = generate_hexagonal_cluster()
    configs.append(state_to_params(c1, a1, 0.0))

    # Config 2: Perturbed angles on hex lattice
    a2 = [30.0 + random.uniform(-5, 5) for _ in range(12)]
    configs.append(state_to_params(c1, a2, 15.0))

    # Config 3: 3x4 rectangular grid
    c3, a3 = generate_rect3x4()
    configs.append(state_to_params(c3, a3, 0.0))

    # Config 4: Two hex-6 clusters
    c4, a4 = generate_hex6_plus6()
    configs.append(state_to_params(c4, a4, 0.0))

    # Config 5: Perturbed rect grid
    c5, a5 = generate_rect3x4()
    a5p = [30.0 + random.uniform(-8, 8) for _ in range(12)]
    c5p = [[c[0] + random.uniform(-0.15, 0.15), c[1] + random.uniform(-0.15, 0.15)] for c in c5]
    configs.append(state_to_params(c5p, a5p, 10.0))

    # Config 6: Hex lattice with random ring-2 selection
    c6, a6 = generate_hexagonal_cluster()
    # Perturb positions slightly
    c6p = [[c[0] + random.uniform(-0.05, 0.05), c[1] + random.uniform(-0.05, 0.05)] for c in c6]
    a6p = [30.0 + random.uniform(-3, 3) for _ in range(12)]
    configs.append(state_to_params(c6p, a6p, 5.0))

    best_params = None
    best_val = float('inf')

    for idx, init_params in enumerate(configs):
        if time.time() > deadline - 40:
            break

        # SLSQP with hard constraints
        p_slsqp, val_slsqp = slsqp_optimize(init_params, max_iter=200)
        centers_s, angles_s, otheta_s = params_to_state(p_slsqp)
        if is_valid(centers_s, angles_s) and val_slsqp < best_val:
            best_val = val_slsqp
            best_params = p_slsqp

        # Nelder-Mead polish
        if time.time() > deadline - 20:
            break
        p_nm, val_nm = nelder_polish(p_slsqp, max_iter=400)
        centers_n, angles_n, otheta_n = params_to_state(p_nm)
        if is_valid(centers_n, angles_n) and val_nm < best_val:
            best_val = val_nm
            best_params = p_nm

    if best_params is None:
        best_params = configs[0]
        centers_b, angles_b, otheta_b = params_to_state(best_params)
        best_val = compute_outer_side(centers_b, angles_b, math.radians(otheta_b))

    # Polish outer angle
    centers_final, angles_final, otheta_final = params_to_state(best_params)
    otheta_final, best_val = polish_outer_angle(centers_final, angles_final, otheta_final)

    # Final Nelder-Mead polish if time remains
    if time.time() < deadline - 12:
        final_params = state_to_params(centers_final, angles_final, otheta_final)
        p_final, val_final = nelder_polish(final_params, max_iter=300)
        centers_f, angles_f, otheta_f = params_to_state(p_final)
        if is_valid(centers_f, angles_f) and val_final < best_val:
            centers_final, angles_final = centers_f, angles_f
            otheta_final, best_val = polish_outer_angle(centers_final, angles_final, otheta_f)

    inner = [[c[0], c[1], a] for c, a in zip(centers_final, angles_final)]
    outer_center = (0.0, 0.0)
    outer_side = best_val
    outer_angle = otheta_final

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
