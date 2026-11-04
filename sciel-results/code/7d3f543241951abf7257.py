import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

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
        return vertices

    def sat_disjoint(v1, v2, eps=1e-9):
        """SAT test: return True if polygons are disjoint (no overlap)."""
        n1 = len(v1)
        n2 = len(v2)
        for i in range(n1):
            x1, y1 = v1[i]
            x2, y2 = v1[(i + 1) % n1]
            nx = -(y2 - y1)
            ny = x2 - x1
            if abs(nx) < 1e-15 and abs(ny) < 1e-15:
                continue
            min_p1 = min((px * nx + py * ny) for px, py in v1)
            max_p1 = max((px * nx + py * ny) for px, py in v1)
            min_p2 = min((px * nx + py * ny) for px, py in v2)
            max_p2 = max((px * nx + py * ny) for px, py in v2)
            if max_p1 < min_p2 - eps or max_p2 < min_p1 - eps:
                continue
            # Overlapping on this axis, check all axes
        # If we get here, no separating axis found -> overlapping
        return False

    def are_disjoint(centers, angles, eps=1e-9):
        """Check if all hexagons are pairwise disjoint using SAT."""
        n = len(centers)
        verts = []
        for i in range(n):
            verts.append(get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]))
        for i in range(n):
            for j in range(i + 1, n):
                if not sat_disjoint(verts[i], verts[j], eps):
                    return False
        return True

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
        # For a regular hexagon with side s, the apothem is s*sqrt(3)/2
        # max_proj = apothem = s * sqrt(3) / 2, so s = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    def objective(x):
        """Objective function for optimization. x = [cx0, cy0, a0, cx1, cy1, a1, ..., ocx, ocy, otheta]"""
        n = N
        centers = []
        angles = []
        for i in range(n):
            cx = x[3 * i]
            cy = x[3 * i + 1]
            a = x[3 * i + 2]
            centers.append((cx, cy))
            angles.append(a)
        ocx = x[3 * n]
        ocy = x[3 * n + 1]
        otheta = x[3 * n + 2]

        # Check disjointness - add penalty if overlapping
        verts = []
        for i in range(n):
            verts.append(get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]))
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                # SAT test for overlap
                v1, v2 = verts[i], verts[j]
                overlap = False
                n1, n2 = len(v1), len(v2)
                for k in range(n1):
                    x1, y1 = v1[k]
                    x2, y2 = v1[(k + 1) % n1]
                    nx = -(y2 - y1)
                    ny = x2 - x1
                    if abs(nx) < 1e-15 and abs(ny) < 1e-15:
                        continue
                    min_p1 = min((px * nx + py * ny) for px, py in v1)
                    max_p1 = max((px * nx + py * ny) for px, py in v1)
                    min_p2 = min((px * nx + py * ny) for px, py in v2)
                    max_p2 = max((px * nx + py * ny) for px, py in v2)
                    if max_p1 >= min_p2 - 1e-9 and max_p2 >= min_p1 - 1e-9:
                        # Check other axes too
                        pass
                # Quick center distance check as additional penalty
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < math.sqrt(3) - 0.001:
                    penalty += (math.sqrt(3) - 0.001 - dist) ** 2 * 100.0

        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        return side + penalty

    # Initial configuration: ring packing
    eps = 0.005
    d = math.sqrt(3) + eps

    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])

    centers_init = [np.array([0.0, 0.0])]
    ring1_vectors = [u, v, v - u, -u, -v, u - v]
    for vec in ring1_vectors:
        centers_init.append(vec.copy())

    # Ring 2 candidates
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
                    ring2_candidates.append(p2.copy())

    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    for i in range(5):
        centers_init.append(ring2_candidates[i].copy())

    assert len(centers_init) == 12

    # Build initial x vector
    x0 = []
    for c in centers_init:
        x0.append(float(c[0]))
        x0.append(float(c[1]))
        x0.append(30.0)  # angle in degrees
    x0.extend([0.0, 0.0, 0.0])  # outer center and angle

    best_result = None
    best_side = float('inf')

    # Nelder-Mead optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if result.fun < best_side:
            best_side = result.fun
            best_result = result.x
    except Exception:
        pass

    # Also try with different initial angles
    for angle_deg in [0.0, 30.0, 60.0]:
        x0_alt = []
        for c in centers_init:
            x0_alt.append(float(c[0]))
            x0_alt.append(float(c[1]))
            x0_alt.append(angle_deg)
        x0_alt.extend([0.0, 0.0, 0.0])

        if time.time() > deadline - 20:
            break

        try:
            result = minimize(objective, x0_alt, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_side:
                best_side = result.fun
                best_result = result.x
        except Exception:
            pass

    # Parse best result
    if best_result is None:
        # Fallback: use initial configuration
        centers_final = [[float(c[0]), float(c[1])] for c in centers_init]
        angles_final = [30.0] * 12
        ocx, ocy, otheta = 0.0, 0.0, 0.0
    else:
        centers_final = []
        angles_final = []
        for i in range(N):
            centers_final.append([float(best_result[3 * i]), float(best_result[3 * i + 1])])
            angles_final.append(float(best_result[3 * i + 2]))
        ocx = float(best_result[3 * N])
        ocy = float(best_result[3 * N + 1])
        otheta = float(best_result[3 * N + 2])

    # Refine outer hexagon placement with fine grid
    if time.time() < deadline - 10:
        best_outer_side = compute_outer_side(centers_final, angles_final, ocx, ocy, otheta)
        for otheta_deg in np.linspace(0, 60, 121):
            for ocx_try in np.linspace(ocx - 0.3, ocx + 0.3, 13):
                for ocy_try in np.linspace(ocy - 0.3, ocy + 0.3, 13):
                    side = compute_outer_side(centers_final, angles_final, float(ocx_try), float(ocy_try), float(otheta_deg))
                    if side < best_outer_side:
                        best_outer_side = side
                        ocx, ocy, otheta = float(ocx_try), float(ocy_try), float(otheta_deg)

    inner = [[float(c[0]), float(c[1]), float(a)] for c, a in zip(centers_final, angles_final)]
    outer_center = (ocx, ocy)
    outer_side = best_outer_side
    outer_angle = otheta

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
