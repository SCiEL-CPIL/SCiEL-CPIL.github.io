import math
import time
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

    def project_minmax(vertices, axis):
        projs = vertices @ axis
        return projs.min(), projs.max()

    def min_distance_hex(v1, v2):
        # SAT: return minimum separation distance (positive = disjoint, negative = overlap)
        # Uses all 12 edge normals + closest-vertex-pair fallback for accuracy
        min_sep = float('inf')
        has_overlap = False

        axes = []
        for i in range(6):
            for v in (v1, v2):
                edge = v[(i + 1) % 6] - v[i]
                axes.append(np.array([-edge[1], edge[0]]))

        for ax in axes:
            ax = ax / (np.linalg.norm(ax) + 1e-15)
            mn1, mx1 = project_minmax(v1, ax)
            mn2, mx2 = project_minmax(v2, ax)
            if mx1 < mn2 or mx2 < mn1:
                sep = max(mn2 - mx1, mn1 - mx2)
                min_sep = min(min_sep, sep)
            else:
                has_overlap = True

        if not has_overlap:
            return min_sep

        # Overlapping: find minimum penetration depth via vertex-edge distances
        min_pen = float('inf')
        for i in range(6):
            for j in range(6):
                # v1[i] to edge of v2 containing v2[j]
                e1 = v2[(j + 1) % 6] - v2[j]
                e2 = v2[j] - v2[(j + 1) % 6]
                # distance from v1[i] to line through v2[j], v2[j+1]
                ex, ey = e1
                d = np.array([v1[i][0] - v2[j][0], v1[i][1] - v2[j][1]])
                t = (d[0] * ex + d[1] * ey) / (ex * ex + ey * ey + 1e-15)
                t = max(0.0, min(1.0, t))
                closest = v2[j] + t * e1
                dist = np.linalg.norm(v1[i] - closest)
                min_pen = min(min_pen, dist)

        return -min_pen

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

    def objective_outer(params):
        ocx, ocy, otheta = params
        return compute_outer_side(centers_fixed, angles_fixed, ocx, ocy, otheta)

    # --- Build exact hexagonal lattice packing ---
    # Regular hexagonal lattice with unit hexagons:
    # Ring 0: center
    # Ring 1: 6 hexagons at distance sqrt(3)
    # Ring 2: 12 hexagons at distance sqrt(3)*2 (we take 5)
    d = math.sqrt(3)
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])

    ring1_vectors = [u, v, v - u, -u, -v, u - v]

    # Generate ring 2 candidates
    ring2_candidates = []
    basis = ring1_vectors
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
    # Take 5 ring-2 hexagons
    selected_ring2 = ring2_candidates[:5]

    centers_fixed = [np.array([0.0, 0.0])]
    for vec in ring1_vectors:
        centers_fixed.append(vec.copy())
    for vec in selected_ring2:
        centers_fixed.append(vec.copy())

    angles_fixed = [30.0] * 12

    # --- Verify disjointness ---
    verts = []
    for i in range(N):
        v = get_hex_vertices(centers_fixed[i][0], centers_fixed[i][1], 1.0, angles_fixed[i])
        verts.append(v)

    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            dist = min_distance_hex(verts[i], verts[j])
            if dist < -1e-9:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: return the lattice packing anyway (shouldn't happen)
        inner = [[c[0], c[1], a] for c, a in zip(centers_fixed, angles_fixed)]
        return inner, (0.0, 0.0), 5.0, 0.0

    # --- Optimize outer hexagon ---
    best_side = 5.0
    best_ocx, best_ocy, best_otheta = 0.0, 0.0, 0.0

    # Try multiple starting rotations for the outer hexagon
    for start_theta in [0.0, 15.0, 30.0, 45.0]:
        if time.time() > deadline - 30:
            break

        x0 = np.array([0.0, 0.0, start_theta])
        try:
            res = minimize(objective_outer, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_side:
                best_side = res.fun
                best_ocx, best_ocy, best_otheta = res.x[0], res.x[1], res.x[2]
        except Exception:
            pass

    # Fine grid polish
    for da in np.linspace(-3, 3, 25):
        for dx in np.linspace(-0.1, 0.1, 7):
            for dy in np.linspace(-0.1, 0.1, 7):
                s = compute_outer_side(centers_fixed, angles_fixed,
                                       best_ocx + dx, best_ocy + dy, best_otheta + da)
                if s < best_side:
                    best_side = s
                    best_ocx += dx
                    best_ocy += dy
                    best_otheta += da

    inner = [[c[0], c[1], a] for c, a in zip(centers_fixed, angles_fixed)]
    outer_center = (best_ocx, best_ocy)
    outer_side = best_side
    outer_angle = best_otheta

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
