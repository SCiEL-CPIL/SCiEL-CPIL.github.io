import numpy as np
import time
import itertools
from scipy.optimize import minimize, minimize_scalar

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    eps = 1e-12

    # Base vertices of unit hexagon, pointy-right (angle 0)
    base_vertices = np.array(
        [[np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)]
    )

    def rot_mat(phi):
        c, s = np.cos(phi), np.sin(phi)
        return np.array([[c, -s], [s, c]])

    def all_vertices(centers, phis):
        """Return (66, 2) array of all inner-hexagon vertices."""
        n_hex = len(centers)
        out = np.empty((n_hex * 6, 2))
        for i in range(n_hex):
            R = rot_mat(phis[i])
            out[i * 6:(i + 1) * 6] = base_vertices @ R.T + centers[i]
        return out

    def convex_hull(pts):
        """Andrew's monotone chain. Returns (m,2) hull in CCW order (no dup last)."""
        pts = np.array(pts, dtype=float)
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        n = len(pts)
        if n <= 2:
            return pts
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = np.array(lower[:-1] + upper[:-1])
        return hull

    def outer_normals(theta_out):
        return np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])

    def compute_outer_side(theta_out, centers, phis):
        """Exact minimum outer side length via convex hull of all inner vertices."""
        verts = all_vertices(centers, phis)
        hull = convex_hull(verts)
        normals = outer_normals(theta_out)
        dots = hull @ normals.T          # (m, 6)
        max_proj = float(np.max(dots))
        return 2.0 * max_proj / np.sqrt(3)

    def objective(params, centers):
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(theta_out, centers, phis)

    def smooth_objective(params, centers, t=80.0):
        theta_out = params[0]
        phis = params[1:]
        verts = all_vertices(centers, phis)
        hull = convex_hull(verts)
        normals = outer_normals(theta_out)
        dots = hull @ normals.T
        max_per_normal = np.max(dots, axis=0)   # (6,)
        m = np.max(max_per_normal)
        lse = m + np.log(np.sum(np.exp(max_per_normal - m)))
        return 2.0 * (lse / t) / np.sqrt(3)

    # All 15 combinations of 4 outer-ring positions out of 6
    all_angles = [0, 60, 120, 180, 240, 300]
    combinations = list(itertools.combinations(all_angles, 4))

    best_score = float('inf')
    best_params = None
    best_centers = None
    best_theta_out = 0.0

    # ---- Phase 1: For each of the 15 topologies, run multi-start Powell ----
    for combo in combinations:
        if time.time() > deadline - 30:
            break

        # Build centers for this topology
        centers = np.array([[0.0, 0.0]])

        # Inner ring (6) at 30, 90, 150, 210, 270, 330 degrees
        r1 = np.sqrt(3) + eps
        for k in range(6):
            angle = np.deg2rad(30 + k * 60)
            centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])

        # Outer ring (4) at chosen angles
        r2 = 3.0 + 2 * eps
        for angle_deg in combo:
            angle = np.deg2rad(angle_deg)
            centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])

        n_hex = len(centers)
        assert n_hex == 11

        # Multi-start Powell for this topology
        n_starts = 6
        for i in range(n_starts):
            if time.time() > deadline - 25:
                break
            if i == 0:
                x_start = np.zeros(n_hex + 1)
            elif i == 1:
                x_start = np.array([np.pi / 6] + [0.0] * n_hex)
            elif i == 2:
                x_start = np.array([0.0] + [np.pi / 6] * n_hex)
            elif i == 3:
                x_start = np.array([np.pi / 6] + [np.pi / 6] * n_hex)
            elif i == 4:
                x_start = np.array([0.0] + [np.pi / 12] * n_hex)
            else:
                x_start = np.random.uniform(0, np.pi / 3, n_hex + 1)
            try:
                res = minimize(lambda p: objective(p, centers), x_start, method='Powell',
                               options={'maxiter': 300, 'disp': False,
                                        'ftol': 1e-14, 'xtol': 1e-14})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
                    best_centers = centers.copy()
                    best_theta_out = res.x[0]
            except Exception:
                pass

    if best_centers is None:
        return _fallback_solution()

    # ---- Phase 2: Nelder-Mead polish with smooth objective ----
    if time.time() < deadline - 15:
        try:
            res2 = minimize(lambda p: smooth_objective(p, best_centers), best_params, method='Nelder-Mead',
                            options={'xatol': 1e-12, 'fatol': 1e-14,
                                     'maxiter': 6000, 'disp': False})
            true_score = objective(res2.x, best_centers)
            if true_score < best_score:
                best_score = true_score
                best_params = res2.x.copy()
        except Exception:
            pass

    # ---- Phase 3: Coordinate descent with fine steps ----
    if time.time() < deadline - 5:
        for iteration in range(60):
            if time.time() > deadline - 3:
                break
            improved = False
            for j in range(len(best_params)):
                if time.time() > deadline - 3:
                    break
                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val
                for delta in [1e-8, -1e-8, 1e-7, -1e-7, 1e-6, -1e-6,
                              1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3,
                              1e-2, -1e-2, 5e-3, -5e-3, 2e-2, -2e-2]:
                    test_val = current_val + delta
                    if test_val < -1e-9 or test_val > np.pi / 3 + 1e-9:
                        continue
                    test_param = best_params.copy()
                    test_param[j] = test_val
                    try:
                        test_score = objective(test_param, best_centers)
                        if test_score < best_local:
                            best_local = test_score
                            best_local_val = test_val
                    except Exception:
                        continue
                if best_local < best_score - 1e-15:
                    best_params[j] = best_local_val
                    best_score = best_local
                    improved = True
            if not improved:
                break

    theta_out = best_params[0]
    phis = best_params[1:]

    inner = []
    for i in range(len(best_centers)):
        inner.append([float(best_centers[i][0]), float(best_centers[i][1]),
                      float(np.rad2deg(phis[i]))])

    side = best_score + 1e-9
    return inner, (0.0, 0.0), float(side), float(np.rad2deg(theta_out))


def _fallback_solution():
    eps = 1e-12
    centers = [(0.0, 0.0)]
    r1 = np.sqrt(3) + eps
    for k in range(6):
        a = np.deg2rad(30 + k * 60)
        centers.append((r1 * np.cos(a), r1 * np.sin(a)))
    r2 = 3.0 + 2 * eps
    for k in range(4):
        a = np.deg2rad(k * 60)
        centers.append((r2 * np.cos(a), r2 * np.sin(a)))

    inner = [[x, y, 0.0] for x, y in centers]

    # Compute side length with all rotations = 0 and outer rotation = 0
    base_v = np.array([[np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))]
                       for k in range(6)])
    normals = np.array([
        [np.cos(np.deg2rad(30 + k * 60)),
         np.sin(np.deg2rad(30 + k * 60))]
        for k in range(6)
    ])

    max_proj = 0.0
    for x, y in centers:
        verts = base_v + np.array([x, y])
        dots = verts @ normals.T
        max_proj = max(max_proj, float(np.max(dots)))

    side = 2.0 * max_proj / np.sqrt(3) + 1e-9
    return inner, (0.0, 0.0), float(side), 0.0


if __name__ == "__main__":
    construct()
