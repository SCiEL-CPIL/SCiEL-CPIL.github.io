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
    np.random.seed(42)

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

    # Vectorized outer-side computation: precompute inner vertex offsets once,
    # then per outer orientation compute max over (outer normals x inner offsets).
    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        otheta_rad = math.radians(otheta_deg)
        # inner vertex offsets (12 x 6 x 2)
        n = len(centers)
        a_rad = np.radians(np.array(angles))
        vi = np.arange(6)
        va = a_rad[:, None] + vi[None, :] * (math.pi / 3)
        offs = np.stack([np.cos(va), np.sin(va)], axis=-1)  # (n,6,2)
        # outer normals (6 x 2)
        na = otheta_rad + math.pi / 6 + np.arange(6) * (math.pi / 3)
        nn = np.stack([np.cos(na), np.sin(na)], axis=-1)  # (6,2)
        # projection of each inner offset onto each outer normal
        # proj[k, i, vi] = sum_d offs[i,vi,d]*nn[k,d]
        proj = np.einsum('vd,ivd->kiv', nn, offs)  # (6, n, 6)
        # add center contribution: (cx-ocx)*nx + (cy-ocy)*ny
        cx = np.array([c[0] for c in centers]) - ocx
        cy = np.array([c[1] for c in centers]) - ocy
        center_proj = np.einsum('kd,i d->ki', nn, np.stack([cx, cy], axis=-1))  # (6, n)
        proj = proj + center_proj[:, :, None]
        max_proj = proj.max()
        return 2.0 * max_proj / math.sqrt(3)

    def quick_disjoint_check(centers, angles, min_dist_sq=3.0 - 0.0002):
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx * dx + dy * dy
                if dist_sq < min_dist_sq:
                    return False
        return True

    def calculate_cost(centers, angles, ocx, ocy, otheta_deg, fast=True):
        side = compute_outer_side(centers, angles, ocx, ocy, otheta_deg)
        if fast:
            if not quick_disjoint_check(centers, angles):
                return side + 50.0
        else:
            verts = []
            for i in range(12):
                v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                verts.append(v)
            for i in range(12):
                for j in range(i + 1, 12):
                    if not are_disjoint_sat(verts[i], verts[j]):
                        return side + 100.0
        return side

    def polish_outer(centers, angles, ocx, ocy, otheta_deg):
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta_deg)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta_deg

        # Multi-scale coordinate + angle descent (angle in degrees)
        for scale_p in [0.5, 0.1, 0.01, 0.001, 0.0001]:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for dx in [-scale_p, 0, scale_p]:
                    for dy in [-scale_p, 0, scale_p]:
                        for da in [i * (60.0 / 60) - 30 for i in range(61)]:
                            s = compute_outer_side(centers, angles,
                                                   best_ocx + dx, best_ocy + dy,
                                                   (best_otheta + da) % 60)
                            if s < best_side - 1e-14:
                                best_side = s
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta = (best_otheta + da) % 60
                                improved = True

        # Fine angle-only sweep
        for _ in range(3):
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for da in [i * (60.0 / 360) - 30 for i in range(361)]:
                    s = compute_outer_side(centers, angles,
                                           best_ocx, best_ocy,
                                           (best_otheta + da) % 60)
                    if s < best_side - 1e-15:
                        best_side = s
                        best_otheta = (best_otheta + da) % 60
                        improved = True

        return best_ocx, best_ocy, best_otheta, best_side

    def local_polish_inner(centers, angles, ocx, ocy, otheta_deg):
        n = len(centers)

        def flatten(centers, angles):
            v = []
            for i in range(n):
                v.extend([centers[i][0], centers[i][1], angles[i]])
            return v

        def unflatten(v):
            c = []
            a = []
            for i in range(n):
                c.append([v[3 * i], v[3 * i + 1]])
                a.append(v[3 * i + 2])
            return c, a

        def objective(v):
            if time.time() > deadline - 5:
                return 1e6
            c, a = unflatten(v)
            return calculate_cost(c, a, ocx, ocy, otheta_deg, fast=True)

        x0 = flatten(centers, angles)

        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < 1e5:
                c, a = unflatten(result.x)
                if quick_disjoint_check(c, a):
                    return c, a
        except Exception:
            pass
        return centers, angles

    def slsqp_optimize(centers, angles, ocx, ocy, otheta_deg, max_iters=200):
        n = len(centers)

        def flatten(centers, angles, ocx, ocy, otheta):
            v = []
            for i in range(n):
                v.extend([centers[i][0], centers[i][1], angles[i]])
            v.extend([ocx, ocy, otheta])
            return v

        def unflatten(v):
            c = []
            a = []
            for i in range(n):
                c.append([v[3 * i], v[3 * i + 1]])
                a.append(v[3 * i + 2])
            ocx, ocy, otheta = v[3 * n], v[3 * n + 1], v[3 * n + 2]
            return c, a, ocx, ocy, otheta

        def objective(v):
            if time.time() > deadline - 10:
                return 1e8
            c, a, ocx, ocy, otheta = unflatten(v)
            return compute_outer_side(c, a, ocx, ocy, otheta)

        constraints = []
        min_dist = math.sqrt(3) - 0.0001
        min_dist_sq = min_dist ** 2
        for i in range(n):
            for j in range(i + 1, n):
                def dist_con(v, i=i, j=j):
                    xi, yi = v[3 * i], v[3 * i + 1]
                    xj, yj = v[3 * j], v[3 * j + 1]
                    return (xi - xj) ** 2 + (yi - yj) ** 2 - min_dist_sq
                constraints.append({'type': 'ineq', 'fun': dist_con})

        bounds = []
        for i in range(n):
            bounds.extend([(-8, 8), (-8, 8), (0, 60)])
        bounds.extend([(-3, 3), (-3, 3), (0, 60)])

        x0 = flatten(centers, angles, ocx, ocy, otheta_deg)

        try:
            result = minimize(objective, x0, method='SLSQP',
                             constraints=constraints, bounds=bounds,
                             options={'maxiter': max_iters, 'ftol': 1e-12})
            if result.fun < 1e6:
                c, a, ocx, ocy, otheta = unflatten(result.x)
                if quick_disjoint_check(c, a):
                    return c, a, ocx, ocy, otheta
        except Exception:
            pass
        return centers, angles, ocx, ocy, otheta_deg

    def optimize_sa(centers_init, angles_init, ocx_init, ocy_init, otheta_init, max_time=60):
        T = 1.0
        T_min = 0.0005
        alpha = 0.997
        steps_per_temp = 30
        start_time = time.time()

        current_centers = [c[:] for c in centers_init]
        current_angles = angles_init[:]
        current_ocx = ocx_init
        current_ocy = ocy_init
        current_otheta = otheta_init

        current_cost = calculate_cost(current_centers, current_angles,
                                      current_ocx, current_ocy, current_otheta, fast=True)

        best_centers = [c[:] for c in current_centers]
        best_angles = current_angles[:]
        best_ocx = current_ocx
        best_ocy = current_ocy
        best_otheta = current_otheta
        best_cost = current_cost

        while T > T_min:
            for _ in range(steps_per_temp):
                if time.time() > start_time + max_time or time.time() > deadline - 15:
                    break

                new_centers = [c[:] for c in current_centers]
                new_angles = current_angles[:]
                new_ocx = current_ocx
                new_ocy = current_ocy
                new_otheta = current_otheta

                i = random.randint(0, 11)
                step_size = T * 0.4
                new_centers[i][0] += random.gauss(0, step_size)
                new_centers[i][1] += random.gauss(0, step_size)
                new_angles[i] = (new_angles[i] + random.gauss(0, step_size * 8)) % 60

                if random.random() < 0.08:
                    new_ocx += random.gauss(0, T * 0.05)
                    new_ocy += random.gauss(0, T * 0.05)
                    new_otheta = (new_otheta + random.gauss(0, T * 0.5)) % 60

                new_cost = calculate_cost(new_centers, new_angles,
                                          new_ocx, new_ocy, new_otheta, fast=True)

                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_centers = new_centers
                    current_angles = new_angles
                    current_ocx = new_ocx
                    current_ocy = new_ocy
                    current_otheta = new_otheta
                    current_cost = new_cost

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_centers = [c[:] for c in current_centers]
                        best_angles = current_angles[:]
                        best_ocx = current_ocx
                        best_ocy = current_ocy
                        best_otheta = current_otheta

            T *= alpha
            if time.time() > start_time + max_time or time.time() > deadline - 15:
                break

        return best_centers, best_angles, best_ocx, best_ocy, best_otheta, best_cost

    def verify_validity(centers, angles):
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint_sat(verts[i], verts[j]):
                    return False
        return True

    # Generate initial configurations using hexagonal lattice
    d = math.sqrt(3) + 0.005
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])

    basis = [u, v, v - u, -u, -v, u - v]
    ring1_vectors = [u, v, v - u, -u, -v, u - v]

    ring2_candidates = []
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

    best_result = None
    best_side = float('inf')

    # Try multiple configurations with different ring2 selections
    num_configs = 6
    time_per_config = max(10, (deadline - time.time() - 30) / num_configs)

    for config_idx in range(num_configs):
        if time.time() > deadline - 35:
            break

        if config_idx == 0:
            ring2_sel = ring2_candidates[:5]
        elif config_idx == 1:
            ring2_sel = ring2_candidates[1:6]
        elif config_idx == 2:
            ring2_sel = ring2_candidates[2:7]
        elif config_idx == 3:
            ring2_sel = ring2_candidates[0:3] + ring2_candidates[6:8]
        elif config_idx == 4:
            ring2_sel = ring2_candidates[3:8]
        else:
            random.seed(config_idx * 777)
            idxs = list(range(len(ring2_candidates)))
            random.shuffle(idxs)
            ring2_sel = [ring2_candidates[i] for i in idxs[:5]]

        centers_init = [[0.0, 0.0]]
        for vec in ring1_vectors:
            centers_init.append([float(vec[0]), float(vec[1])])
        for vec in ring2_sel:
            centers_init.append([float(vec[0]), float(vec[1])])

        angles_init = [30.0] * 12
        ocx_init = 0.0
        ocy_init = 0.0
        otheta_init = 0.0

        if not quick_disjoint_check(centers_init, angles_init):
            continue

        # Run SA
        sa_centers, sa_angles, sa_ocx, sa_ocy, sa_otheta, sa_cost = optimize_sa(
            centers_init, angles_init, ocx_init, ocy_init, otheta_init,
            max_time=time_per_config * 0.4
        )

        # SLSQP optimization
        slq_centers, slq_angles, slq_ocx, slq_ocy, slq_otheta = slsqp_optimize(
            sa_centers, sa_angles, sa_ocx, sa_ocy, sa_otheta,
            max_iters=100
        )

        # Local polish inner positions
        pol_centers, pol_angles = local_polish_inner(
            slq_centers, slq_angles, slq_ocx, slq_ocy, slq_otheta
        )

        # Polish outer hexagon
        pol_ocx, pol_ocy, pol_otheta, pol_side = polish_outer(
            pol_centers, pol_angles, slq_ocx, slq_ocy, slq_otheta
        )

        # Verify full validity
        if verify_validity(pol_centers, pol_angles):
            final_side = compute_outer_side(pol_centers, pol_angles,
                                            pol_ocx, pol_ocy, pol_otheta)
            if final_side < best_side:
                best_side = final_side
                best_result = (pol_centers, pol_angles, pol_ocx, pol_ocy, pol_otheta)

        # Also check SA result directly
        if verify_validity(sa_centers, sa_angles):
            sa_side = compute_outer_side(sa_centers, sa_angles,
                                         sa_ocx, sa_ocy, sa_otheta)
            if sa_side < best_side:
                best_side = sa_side
                best_result = (sa_centers, sa_angles, sa_ocx, sa_ocy, sa_otheta)

    # Fallback: use the initial lattice configuration
    if best_result is None:
        centers_init = [[0.0, 0.0]]
        for vec in ring1_vectors:
            centers_init.append([float(vec[0]), float(vec[1])])
        for vec in ring2_candidates[:5]:
            centers_init.append([float(vec[0]), float(vec[1])])
        angles_init = [30.0] * 12
        best_result = (centers_init, angles_init, 0.0, 0.0, 0.0)
        best_side = compute_outer_side(centers_init, angles_init, 0.0, 0.0, 0.0)

    best_centers, best_angles, best_ocx, best_ocy, best_otheta = best_result

    # Final outer polish using the POLISHED centers (bug fix: use best_centers, not originals)
    best_ocx, best_ocy, best_otheta, best_side = polish_outer(
        best_centers, best_angles, best_ocx, best_ocy, best_otheta
    )

    inner = [[c[0], c[1], a] for c, a in zip(best_centers, best_angles)]
    outer_center = (best_ocx, best_ocy)
    outer_side = best_side
    outer_angle = best_otheta

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
