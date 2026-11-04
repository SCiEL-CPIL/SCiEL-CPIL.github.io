import math
import time
import random
import numpy as np
from scipy.optimize import minimize, Bounds

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
        return vertices

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        """Compute minimum outer hexagon side for given configuration."""
        otheta = math.radians(otheta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def check_disjoint_centers(centers, min_dist=math.sqrt(3) - 0.0001):
        """Check if all hexagons are disjoint using center distance (valid for aligned hexagons)."""
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    return False
        return True

    def sat_disjoint(hex1, hex2, eps=1e-6):
        """Check if two hexagons are disjoint using Separating Axis Theorem."""
        x1, y1, a1 = hex1
        x2, y2, a2 = hex2
        verts1 = get_hex_vertices(x1, y1, 1.0, a1)
        verts2 = get_hex_vertices(x2, y2, 1.0, a2)

        axes = []
        for verts in [verts1, verts2]:
            for i in range(6):
                x_a, y_a = verts[i]
                x_b, y_b = verts[(i + 1) % 6]
                dx = x_b - x_a
                dy = y_b - y_a
                length = math.sqrt(dx * dx + dy * dy)
                if length > 1e-10:
                    nx = -dy / length
                    ny = dx / length
                    axes.append((nx, ny))

        for nx, ny in axes:
            min1, max1 = float('inf'), float('-inf')
            min2, max2 = float('inf'), float('-inf')
            for px, py in verts1:
                proj = px * nx + py * ny
                min1 = min(min1, proj)
                max1 = max(max1, proj)
            for px, py in verts2:
                proj = px * nx + py * ny
                min2 = min(min2, proj)
                max2 = max(max2, proj)
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        return False

    def hexagons_disjoint(inner, eps=1e-6):
        """Check if all hexagons are disjoint using SAT."""
        n = len(inner)
        for i in range(n):
            for j in range(i + 1, n):
                if not sat_disjoint(inner[i], inner[j], eps):
                    return False
        return True

    def polish_outer(centers, angles, initial_ocx, initial_ocy, initial_otheta):
        """High-precision polish of outer hexagon parameters using multi-scale search."""
        best_side = compute_outer_side(centers, angles, initial_ocx, initial_ocy, initial_otheta)
        best_ocx, best_ocy, best_otheta = initial_ocx, initial_ocy, initial_otheta

        scales = [0.5, 0.1, 0.01, 0.001, 0.0001]
        for scale in scales:
            if time.time() > deadline - 10.0:
                break
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da_deg in [-scale * 20, 0, scale * 20]:
                            new_otheta = (best_otheta + da_deg) % 60
                            side = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, new_otheta)
                            if side < best_side - 1e-12:
                                best_side = side
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta = new_otheta
                                improved = True

        return best_ocx, best_ocy, best_otheta, best_side

    def optimize_with_slsqp(centers, angles, max_iters=200):
        """Use SLSQP to optimize positions and outer hexagon parameters with hard distance constraints."""
        n = len(centers)

        x0 = []
        for i in range(n):
            x0.extend([centers[i][0], centers[i][1], angles[i]])
        x0.extend([0.0, 0.0, 0.0])  # outer center and angle

        def objective(x):
            if time.time() > deadline - 5.0:
                return 1e10
            c = [[x[3 * i], x[3 * i + 1]] for i in range(n)]
            a = [x[3 * i + 2] for i in range(n)]
            ocx, ocy, otheta = x[3 * n], x[3 * n + 1], x[3 * n + 2]
            side = compute_outer_side(c, a, ocx, ocy, otheta)
            return side

        # Constraints: pairwise center distance >= sqrt(3)
        constraints = []
        min_d = math.sqrt(3) - 0.0001
        min_d_sq = min_d * min_d
        for i in range(n):
            for j in range(i + 1, n):
                def dist_constraint(x, i=i, j=j):
                    xi = x[3 * i]
                    yi = x[3 * i + 1]
                    xj = x[3 * j]
                    yj = x[3 * j + 1]
                    return (xi - xj) ** 2 + (yi - yj) ** 2 - min_d_sq
                constraints.append({'type': 'ineq', 'fun': dist_constraint})

        bounds = []
        for i in range(n):
            bounds.extend([(-6, 6), (-6, 6), (0, 60)])
        bounds.extend([(-3, 3), (-3, 3), (0, 60)])

        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                              constraints=constraints,
                              options={'maxiter': max_iters, 'ftol': 1e-10, 'maxls': 50})
            if result.success or result.fun < 1e9:
                c = [[result.x[3 * i], result.x[3 * i + 1]] for i in range(n)]
                a = [result.x[3 * i + 2] for i in range(n)]
                ocx, ocy, otheta = result.x[3 * n], result.x[3 * n + 1], result.x[3 * n + 2]
                # Verify with SAT
                inner = [[c[i][0], c[i][1], a[i]] for i in range(n)]
                if hexagons_disjoint(inner):
                    side = compute_outer_side(c, a, ocx, ocy, otheta)
                    return c, a, ocx, ocy, otheta, side
        except Exception:
            pass

        return None

    def local_polish(centers, angles, ocx, ocy, otheta, max_iter=300, step_size=0.02):
        """Local search polish with SAT validation."""
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        best_centers = [list(c) for c in centers]
        best_angles = list(angles)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta

        current_centers = [list(c) for c in centers]
        current_angles = list(angles)
        current_ocx, current_ocy, current_otheta = ocx, ocy, otheta
        current_side = best_side

        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            improved = False
            for _ in range(30):
                new_centers = [list(c) for c in current_centers]
                new_angles = list(current_angles)
                new_ocx, new_ocy, new_otheta = current_ocx, current_ocy, current_otheta

                r = random.random()
                if r < 0.7:
                    idx = random.randint(0, len(new_centers) - 1)
                    dx = random.uniform(-step_size, step_size)
                    dy = random.uniform(-step_size, step_size)
                    new_centers[idx][0] += dx
                    new_centers[idx][1] += dy
                elif r < 0.85:
                    dx = random.uniform(-step_size, step_size)
                    dy = random.uniform(-step_size, step_size)
                    new_ocx += dx
                    new_ocy += dy
                else:
                    dtheta = random.uniform(-1.0, 1.0)
                    new_otheta += dtheta

                # Check disjointness with SAT
                inner = [[new_centers[i][0], new_centers[i][1], new_angles[i]] for i in range(N)]
                if not hexagons_disjoint(inner):
                    continue

                new_side = compute_outer_side(new_centers, new_angles, new_ocx, new_ocy, new_otheta)

                if new_side < current_side - 1e-10:
                    current_centers = new_centers
                    current_angles = new_angles
                    current_ocx, current_ocy, current_otheta = new_ocx, new_ocy, new_otheta
                    current_side = new_side
                    improved = True
                    step_size *= 0.97
                    break

            if not improved:
                step_size *= 0.85
                if step_size < 0.0005:
                    break

            if current_side < best_side - 1e-12:
                best_side = current_side
                best_centers = [list(c) for c in current_centers]
                best_angles = list(current_angles)
                best_ocx, best_ocy, best_otheta = current_ocx, current_ocy, current_otheta

        return best_centers, best_angles, best_ocx, best_ocy, best_otheta, best_side

    # Generate multiple initial configurations
    eps = 0.001
    d = math.sqrt(3) + eps

    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])

    basis = [u, v, v - u, -u, -v, u - v]

    ring1_vectors = [u, v, v - u, -u, -v, u - v]

    # Generate all ring 2 candidates
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

    # Try multiple ring-2 selections
    num_configs = min(4, len(ring2_candidates))

    for config_idx in range(num_configs):
        if time.time() > deadline - 30.0:
            break

        if config_idx == 0:
            ring2_sel = ring2_candidates[:5]
        elif config_idx == 1:
            ring2_sel = [ring2_candidates[i] for i in [0, 1, 3, 5, 7] if i < len(ring2_candidates)]
            if len(ring2_sel) < 5:
                continue
        elif config_idx == 2:
            ring2_sel = [ring2_candidates[i] for i in [1, 2, 4, 6, 8] if i < len(ring2_candidates)]
            if len(ring2_sel) < 5:
                continue
        else:
            random.seed(config_idx * 777)
            idxs = list(range(len(ring2_candidates)))
            random.shuffle(idxs)
            ring2_sel = [ring2_candidates[i] for i in idxs[:5]]

        centers = [[0.0, 0.0]]
        for vec in ring1_vectors:
            centers.append([vec[0], vec[1]])
        for vec in ring2_sel:
            centers.append([vec[0], vec[1]])

        angles = [30.0] * 12

        if not check_disjoint_centers(centers):
            continue

        # SLSQP optimization
        slsqp_result = optimize_with_slsqp(centers, angles, max_iters=150)

        if slsqp_result is not None:
            c, a, ocx, ocy, otheta, side = slsqp_result
            if side < best_side:
                best_side = side
                best_result = (c, a, ocx, ocy, otheta)

    # Fallback: use ring12 with larger epsilon
    if best_result is None:
        eps = 0.05
        d = math.sqrt(3) + eps
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        centers = [[0.0, 0.0]]
        for vec in ring1_vectors:
            centers.append([vec[0], vec[1]])
        basis = [u, v, v - u, -u, -v, u - v]
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
        for i in range(5):
            centers.append([ring2_candidates[i][0], ring2_candidates[i][1]])
        angles = [30.0] * 12
        best_result = (centers, angles, 0.0, 0.0, 0.0)
        best_side = compute_outer_side(centers, angles, 0.0, 0.0, 0.0)

    centers, angles, ocx, ocy, otheta = best_result

    # Local polish
    if time.time() < deadline - 15.0:
        centers, angles, ocx, ocy, otheta, best_side = local_polish(
            centers, angles, ocx, ocy, otheta, max_iter=200, step_size=0.02
        )

    # High-precision outer hexagon polish
    if time.time() < deadline - 5.0:
        ocx, ocy, otheta, best_side = polish_outer(centers, angles, ocx, ocy, otheta)

    inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
    outer_center = (ocx, ocy)
    outer_side = best_side
    outer_angle = otheta

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
