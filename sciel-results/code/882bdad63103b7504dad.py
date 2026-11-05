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
        dots = hull @ normals.T
        max_proj = float(np.max(dots))
        return 2.0 * max_proj / np.sqrt(3)

    def objective(params):
        """
        params layout:
          [0] = theta_out (outer hexagon rotation)
          [1:23] = 11 * 2 = 22 center coords (x0,y0,x1,y1,...,x10,y10)
          [23:34] = 11 rotation angles phi_i
        """
        theta_out = params[0]
        centers = params[1:23].reshape(11, 2)
        phis = params[23:34]
        return compute_outer_side(theta_out, centers, phis)

    def smooth_objective(params, t=60.0):
        """Smooth surrogate using log-sum-exp over the 6 face projections."""
        theta_out = params[0]
        centers = params[1:23].reshape(11, 2)
        phis = params[23:34]
        verts = all_vertices(centers, phis)
        hull = convex_hull(verts)
        normals = outer_normals(theta_out)
        dots = hull @ normals.T
        max_per_normal = np.max(dots, axis=0)
        m = np.max(max_per_normal)
        lse = m + np.log(np.sum(np.exp(max_per_normal - m)))
        return 2.0 * (lse / t) / np.sqrt(3)

    # All 15 combinations of 4 outer-ring positions out of 6
    all_angles = [0, 60, 120, 180, 240, 300]
    combinations = list(itertools.combinations(all_angles, 4))

    best_score = float('inf')
    best_params = None

    # ---- Build initial centers for a given combo ----
    def make_centers(combo):
        centers = np.array([[0.0, 0.0]])
        r1 = np.sqrt(3) + eps
        for k in range(6):
            angle = np.deg2rad(30 + k * 60)
            centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        r2 = 3.0 + 2 * eps
        for angle_deg in combo:
            angle = np.deg2rad(angle_deg)
            centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        return centers

    # ---- Phase 1: Multi-start Powell for each topology ----
    n_starts = 4
    for ci, combo in enumerate(combinations):
        if time.time() > deadline - 40:
            break

        centers = make_centers(combo)
        n_hex = len(centers)
        assert n_hex == 11

        for i in range(n_starts):
            if time.time() > deadline - 35:
                break

            if i == 0:
                x_start = np.zeros(34)
            elif i == 1:
                x_start = np.zeros(34)
                x_start[0] = np.pi / 6
            elif i == 2:
                x_start = np.zeros(34)
                x_start[23:] = np.pi / 6
            else:
                x_start = np.random.uniform(-0.1, 0.1, 34)
                x_start[0] = np.random.uniform(0, np.pi / 6)

            try:
                res = minimize(objective, x_start, method='Powell',
                               options={'maxiter': 200, 'disp': False,
                                        'ftol': 1e-12, 'xtol': 1e-12})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    if best_params is None:
        return _fallback_solution()

    # ---- Phase 2: Nelder-Mead polish with smooth objective ----
    if time.time() < deadline - 20:
        try:
            res2 = minimize(smooth_objective, best_params, method='Nelder-Mead',
                            options={'xatol': 1e-10, 'fatol': 1e-12,
                                     'maxiter': 4000, 'disp': False})
            true_score = objective(res2.x)
            if true_score < best_score:
                best_score = true_score
                best_params = res2.x.copy()
        except Exception:
            pass

    # ---- Phase 3: Coordinate descent with fine steps ----
    if time.time() < deadline - 8:
        for iteration in range(80):
            if time.time() > deadline - 5:
                break
            improved = False
            for j in range(len(best_params)):
                if time.time() > deadline - 5:
                    break
                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val
                for delta in [1e-8, -1e-8, 1e-7, -1e-7, 1e-6, -1e-6,
                              1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3,
                              5e-3, -5e-3, 1e-2, -1e-2, 2e-2, -2e-2]:
                    test_val = current_val + delta
                    # Bounds: theta_out in [0, pi/6], centers in [-5,5], phis in [0, pi/3]
                    if j == 0:
                        if test_val < -1e-9 or test_val > np.pi / 6 + 1e-9:
                            continue
                    elif j < 23:
                        if test_val < -5.0 or test_val > 5.0:
                            continue
                    else:
                        if test_val < -1e-9 or test_val > np.pi / 3 + 1e-9:
                            continue
                    test_param = best_params.copy()
                    test_param[j] = test_val
                    try:
                        test_score = objective(test_param)
                        if test_score < best_local:
                            best_local = test_score
                            best_local_val = test_val
                    except Exception:
                        continue
                if best_local < best_score - 1e-14:
                    best_params[j] = best_local_val
                    best_score = best_local
                    improved = True
            if not improved:
                break

    # ---- Extract solution ----
    theta_out = best_params[0]
    centers = best_params[1:23].reshape(11, 2)
    phis = best_params[23:34]

    inner = []
    for i in range(11):
        inner.append([float(centers[i][0]), float(centers[i][1]),
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
