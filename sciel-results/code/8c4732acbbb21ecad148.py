import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
SOFT_MIN_ALPHA = 80.0  # sharper soft-min for better overlap penalty

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0

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

    def sat_separated(v1, v2, eps=1e-9):
        """Check if two hexagons are separated using SAT."""
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            nx, ny = -edge[1], edge[0]
            min1, max1 = project_range(v1, nx, ny)
            min2, max2 = project_range(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            nx, ny = -edge[1], edge[0]
            min1, max1 = project_range(v1, nx, ny)
            min2, max2 = project_range(v2, nx, ny)
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

    def is_valid(centers, angles):
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        for i in range(12):
            for j in range(i + 1, 12):
                if not sat_separated(verts[i], verts[j]):
                    return False
        return True

    def compute_min_separation(verts_i, verts_j):
        """Compute minimum separation over all SAT axes for two hexagons."""
        min_sep = float('inf')
        vi, vj = verts_i, verts_j
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
        return centers, angles

    def generate_rect3x4():
        """3x4 rectangular grid of hexagons."""
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
        """Two hexagonal clusters of 6 (center + ring) offset from each other."""
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

    def slsqp_optimize(initial_params, max_iter=300):
        """
        SLSQP with SAT-based non-overlap constraints.
        Parameters: 24 (centers) + 12 (angles) + 1 (outer angle) = 37
        """
        def objective(params):
            if time.time() > deadline - 5:
                return 1e9
            centers = []
            for i in range(12):
                centers.append([params[i * 2], params[i * 2 + 1]])
            angles = [params[24 + i] for i in range(12)]
            otheta = params[36]
            side = compute_outer_side(centers, angles, 0.0, 0.0, otheta)
            return side

        def constraints():
            cons = []
            for i in range(12):
                for j in range(i + 1, 12):
                    def sep_constraint(params, i=i, j=j):
                        if time.time() > deadline - 5:
                            return 1.0
                        ci = [params[i * 2], params[i * 2 + 1]]
                        cj = [params[j * 2], params[j * 2 + 1]]
                        ai = params[24 + i]
                        aj = params[24 + j]
                        vi = get_hex_vertices(ci[0], ci[1], 1.0, ai)
                        vj = get_hex_vertices(cj[0], cj[1], 1.0, aj)
                        min_sep = compute_min_separation(vi, vj)
                        return min_sep - 1e-6
                    cons.append({'type': 'ineq', 'fun': sep_constraint})
            return cons

        try:
            res = minimize(objective, initial_params, method='SLSQP',
                          constraints=constraints(),
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False})
            return res.x, res.fun
        except Exception:
            return initial_params, objective(initial_params)

    def nelder_polish(params, max_iter=600):
        """Nelder-Mead polish with soft overlap penalty."""
        def obj(params):
            if time.time() > deadline - 5:
                return 1e9
            centers = []
            for i in range(12):
                centers.append([params[i * 2], params[i * 2 + 1]])
            angles = [params[24 + i] for i in range(12)]
            otheta = params[36]
            side = compute_outer_side(centers, angles, 0.0, 0.0, otheta)
            
            penalty = 0.0
            verts = []
            for i in range(12):
                v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                verts.append(v)
            for i in range(12):
                for j in range(i + 1, 12):
                    min_sep = compute_min_separation(verts[i], verts[j])
                    if min_sep < 0:
                        # Use soft-min approximation for smoother penalty
                        penalty += math.exp(SOFT_MIN_ALPHA * min_sep) / SOFT_MIN_ALPHA
            return side + 100.0 * penalty

        try:
            res = minimize(obj, params, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
            return res.x, res.fun
        except Exception:
            return params, obj(params)

    # Generate initial configurations
    configs = []
    
    # Config 1: Standard hex lattice (1+6+5 ring)
    c1, a1 = generate_hexagonal_cluster()
    params1 = []
    for cx, cy in c1:
        params1.append(cx)
        params1.append(cy)
    for a in a1:
        params1.append(a)
    params1.append(0.0)
    configs.append(params1)
    
    # Config 2: Perturbed angles on hex lattice
    c2, a2 = c1, [30.0 + random.uniform(-3, 3) for _ in range(12)]
    params2 = []
    for cx, cy in c2:
        params2.append(cx)
        params2.append(cy)
    for a in a2:
        params2.append(a)
    params2.append(15.0)
    configs.append(params2)

    # Config 3: 3x4 rectangular grid
    c3, a3 = generate_rect3x4()
    params3 = []
    for cx, cy in c3:
        params3.append(cx)
        params3.append(cy)
    for a in a3:
        params3.append(a)
    params3.append(0.0)
    configs.append(params3)

    # Config 4: Two hex-6 clusters
    c4, a4 = generate_hex6_plus6()
    params4 = []
    for cx, cy in c4:
        params4.append(cx)
        params4.append(cy)
    for a in a4:
        params4.append(a)
    params4.append(0.0)
    configs.append(params4)

    # Config 5: Perturbed rect grid
    c5, a5 = generate_rect3x4()
    a5p = [30.0 + random.uniform(-5, 5) for _ in range(12)]
    params5 = []
    for cx, cy in c5:
        params5.append(cx + random.uniform(-0.1, 0.1))
        params5.append(cy + random.uniform(-0.1, 0.1))
    for a in a5p:
        params5.append(a)
    params5.append(10.0)
    configs.append(params5)

    # Config 6: Another perturbation of hex cluster with different outer angle
    c6, a6 = generate_hexagonal_cluster()
    a6p = [30.0 + random.uniform(-2, 2) for _ in range(12)]
    params6 = []
    for cx, cy in c6:
        params6.append(cx + random.uniform(-0.05, 0.05))
        params6.append(cy + random.uniform(-0.05, 0.05))
    for a in a6p:
        params6.append(a)
    params6.append(30.0)
    configs.append(params6)

    random.seed(42)
    best_params = None
    best_val = float('inf')

    for idx, init_params in enumerate(configs):
        if time.time() > deadline - 50:
            break
        
        # SLSQP with hard constraints
        p_slsqp, val_slsqp = slsqp_optimize(init_params, max_iter=200)
        if val_slsqp < best_val and is_valid(
            [[p_slsqp[i*2], p_slsqp[i*2+1]] for i in range(12)],
            [p_slsqp[24+i] for i in range(12)]
        ):
            best_val = val_slsqp
            best_params = p_slsqp
        
        # Nelder-Mead polish
        if time.time() > deadline - 30:
            break
        p_nm, val_nm = nelder_polish(p_slsqp, max_iter=500)
        if val_nm < best_val and is_valid(
            [[p_nm[i*2], p_nm[i*2+1]] for i in range(12)],
            [p_nm[24+i] for i in range(12)]
        ):
            best_val = val_nm
            best_params = p_nm

    if best_params is None:
        best_params = configs[0]
        best_val = compute_outer_side(
            [[best_params[i*2], best_params[i*2+1]] for i in range(12)],
            [best_params[24+i] for i in range(12)],
            0.0, 0.0, best_params[36]
        )

    # Final: high-precision grid search on outer angle
    centers = [[best_params[i*2], best_params[i*2+1]] for i in range(12)]
    angles = [best_params[24+i] for i in range(12)]
    best_side = best_val
    best_otheta = best_params[36]
    
    # Coarse search
    for da in np.linspace(-18, 18, 37):
        s = compute_outer_side(centers, angles, 0.0, 0.0, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da

    # High-precision polish pass if time remains
    if time.time() < deadline - 15:
        # Fine-tune outer angle with multi-scale approach
        scales = [3.0, 1.0, 0.3, 0.1, 0.03]
        for scale in scales:
            for da in np.linspace(-scale, scale, int(2*scale/0.05)+1 if scale >= 0.1 else 13):
                s = compute_outer_side(centers, angles, 0.0, 0.0, best_otheta + da)
                if s < best_side:
                    best_side = s
                    best_otheta = best_otheta + da
        
        # Local Nelder-Mead polish on final config with tight tolerance
        final_params = list(best_params)
        final_params[36] = best_otheta
        p_final, val_final = nelder_polish(final_params, max_iter=400)
        if val_final < best_side and is_valid(
            [[p_final[i*2], p_final[i*2+1]] for i in range(12)],
            [p_final[24+i] for i in range(12)]
        ):
            centers = [[p_final[i*2], p_final[i*2+1]] for i in range(12)]
            angles = [p_final[24+i] for i in range(12)]
            best_otheta = p_final[36]
            best_side = val_final
            
            # Final angle refinement
            for scale in [1.0, 0.3, 0.1, 0.03]:
                for da in np.linspace(-scale, scale, int(2*scale/0.05)+1 if scale >= 0.1 else 13):
                    s = compute_outer_side(centers, angles, 0.0, 0.0, best_otheta + da)
                    if s < best_side:
                        best_side = s
                        best_otheta = best_otheta + da

    # Center the configuration for cleanliness
    cx_mean = sum(c[0] for c in centers) / 12
    cy_mean = sum(c[1] for c in centers) / 12
    for c in centers:
        c[0] -= cx_mean
        c[1] -= cy_mean

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
