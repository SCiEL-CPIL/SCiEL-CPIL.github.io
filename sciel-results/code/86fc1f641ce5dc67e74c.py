import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.spatial import ConvexHull

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def get_normals(beta):
        return np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

    def compute_D(pts, center, normals):
        centered = pts - np.array(center)
        projs = centered @ normals.T
        return float(np.max(np.abs(projs)))

    def calc_S_for_beta(pts, beta, center=None):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, float('inf')

        normals = get_normals(beta)
        c_obj = np.array([0.0, 0.0, 1.0])
        n_pts = len(pts)

        A_ub_list = []
        b_ub_list = []
        for nx, ny in normals:
            dots = pts[:, 0] * nx + pts[:, 1] * ny
            A_ub_list.append(np.column_stack([-nx * np.ones(n_pts), -ny * np.ones(n_pts), -np.ones(n_pts)]))
            b_ub_list.append(-dots)
            A_ub_list.append(np.column_stack([nx * np.ones(n_pts), ny * np.ones(n_pts), -np.ones(n_pts)]))
            b_ub_list.append(dots)

        A_ub = np.vstack(A_ub_list)
        b_ub = np.concatenate(b_ub_list)
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = float(res.x[2])
                cx, cy = float(res.x[0]), float(res.x[1])
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, float('inf')
        except Exception:
            return float('inf'), None, float('inf')

    def hexagon_vertices(cx, cy, angle_deg, s=1.0):
        rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            a = rad + k * math.pi / 3.0
            verts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
        return np.array(verts)

    def polygons_disjoint(v1, v2, eps=1e-9):
        # Check if two convex polygons are disjoint using separating axis theorem
        for verts in [v1, v2]:
            for i in range(len(verts)):
                p1 = verts[i]
                p2 = verts[(i + 1) % len(verts)]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                if np.linalg.norm(normal) < 1e-12:
                    continue
                normal /= np.linalg.norm(normal)
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                if np.max(proj1) < np.min(proj2) - eps or np.max(proj2) < np.min(proj1) - eps:
                    return True  # separated, so disjoint
        return False  # no separating axis found, so overlapping

    def check_disjointness(inner_hexes, eps=1e-6):
        n = len(inner_hexes)
        verts_list = [hexagon_vertices(x, y, a) for x, y, a in inner_hexes]
        for i in range(n):
            for j in range(i + 1, n):
                if not polygons_disjoint(verts_list[i], verts_list[j], eps):
                    return False
        return True

    def repulsion_energy(inner_hexes):
        n = len(inner_hexes)
        energy = 0.0
        verts_list = [hexagon_vertices(x, y, a) for x, y, a in inner_hexes]
        for i in range(n):
            for j in range(i + 1, n):
                # Compute minimum distance between two convex polygons
                # Use vertex-to-edge distance
                min_dist = float('inf')
                for vi in verts_list[i]:
                    for vj in verts_list[j]:
                        d = np.linalg.norm(vi - vj)
                        min_dist = min(min_dist, d)
                # Also check edge-to-edge
                for i_edge in range(6):
                    for j_edge in range(6):
                        p1 = verts_list[i][i_edge]
                        p2 = verts_list[i][(i_edge + 1) % 6]
                        p3 = verts_list[j][j_edge]
                        p4 = verts_list[j][(j_edge + 1) % 6]
                        # Line segment distance approximation
                        # For convex hexagons, vertex-vertex is often sufficient
                        pass
                if min_dist < 0.05:
                    energy += 1000.0 / (min_dist + 1e-9)
                elif min_dist < 0.1:
                    energy += 10.0 / (min_dist + 1e-9)
        return energy

    # --- Generate diverse candidate configurations ---
    candidates = []

    # Classic hexagonal cluster
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    # Elongated shapes
    axial_2 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_3)

    # Compact irregular
    axial_4 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    # Star-like
    axial_6 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_7)

    # L-shape
    axial_8 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4), (2, 3)
    ]
    candidates.append(axial_8)

    # T-shape
    axial_9 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_9)

    # Zigzag
    axial_10 = [
        (0, 0), (1, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (1, 3), (2, 3),
        (0, 4), (1, 4), (2, 4)
    ]
    candidates.append(axial_10)

    # Dense block
    axial_11 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_11)

    # Asymmetric
    axial_12 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_12)

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_15)

    # --- Force-directed layout optimization ---
    def force_directed_layout(pts, n_iter=2000, learning_rate=0.01, repulsion_strength=1.0):
        """Optimize positions using force-directed method to maximize separation."""
        n = len(pts)
        positions = pts.copy().astype(float)
        angles = np.zeros(n)  # All start at 0 degrees

        best_positions = positions.copy()
        best_energy = float('inf')

        for it in range(n_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break

            # Compute forces
            forces = np.zeros_like(positions)
            energy = 0.0

            for i in range(n):
                for j in range(i + 1, n):
                    v_i = hexagon_vertices(positions[i, 0], positions[i, 1], math.degrees(angles[i]))
                    v_j = hexagon_vertices(positions[j, 0], positions[j, 1], math.degrees(angles[j]))

                    # Check overlap
                    if not polygons_disjoint(v_i, v_j, 1e-6):
                        # Repulsive force
                        diff = positions[j] - positions[i]
                        dist = np.linalg.norm(diff)
                        if dist < 1e-9:
                            diff = np.array([1e-9, 0.0])
                            dist = 1e-9
                        force = repulsion_strength * diff / (dist * dist)
                        forces[i] -= force
                        forces[j] += force
                        energy += repulsion_strength / (dist + 1e-9)

            # Update positions
            positions += learning_rate * forces
            positions *= 0.999  # Slight damping

            # Track best
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()

        return best_positions, angles

    # --- Optimize a fixed point set ---
    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, best_center

        # Phase 1: Coarse grid over beta
        n_coarse = 100
        for i in range(n_coarse):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_coarse - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                if center is not None:
                    best_center = center

        # Phase 2: Bounded scalar refine
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                lo = max(0.0, best_beta - math.pi / 60)
                hi = min(math.pi / 6, best_beta + math.pi / 60)
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                      bounds=(lo, hi), method='bounded',
                                      options={'maxiter': 500, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = float(res.fun)
                    best_beta = float(res.x)
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Phase 3: Fine grid
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                n_fine = 500
                span = math.pi / 6 / 30
                for i in range(n_fine):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    b = best_beta + (i / (n_fine - 1) - 0.5) * 2 * span
                    b = ((b % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    S, center, D = calc_S_for_beta(pts, b)
                    if S < best_S:
                        best_S = S
                        best_beta = b
                        if center is not None:
                            best_center = center
        except Exception:
            pass

        # Phase 4: Bounded scalar on narrow window
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                lo = max(0.0, best_beta - span / 4)
                hi = min(math.pi / 6, best_beta + span / 4)
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                      bounds=(lo, hi), method='bounded',
                                      options={'maxiter': 2000, 'xatol': 1e-16})
                if res.fun < best_S:
                    best_S = float(res.fun)
                    best_beta = float(res.x)
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        return best_S, best_beta, best_center

    # --- Main loop ---
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # --- Fallback ---
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])),
                              float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        normals = get_normals(best_global_beta)
        max_proj = 0.0
        for nx, ny in normals:
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # --- Final: recompute exact S ---
    normals = get_normals(best_global_beta)
    max_proj = 0.0
    for nx, ny in normals:
        projs = (best_global_pts[:, 0] - cx) * nx + (best_global_pts[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))

    D_final = max_proj
    S_final = (D_final + APOTHEM) / APOTHEM

    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
