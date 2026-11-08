import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    # Safety margin: aim to finish well before 150s
    time_limit = 140.0

    # Precompute triangle indices for vectorization
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """
        Compute min triangle area / convex hull area.
        Returns 0.0 if invalid.
        """
        try:
            pts = np.asarray(points, dtype=np.float64)
            pts = pts - pts.mean(axis=0)
            # Check for degenerate hull
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0

        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)

        if min_area < 0:
            return 0.0
        return min_area / area_hull

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def points_from_radii(radii):
        """
        Generate points from 13 radii with fixed angles 2*pi*k/13.
        """
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        return np.column_stack([xs, ys])

    def objective_symmetric(radii):
        """
        Objective function for symmetric optimization.
        Returns negative score to be minimized.
        """
        pts = points_from_radii(radii)
        score = compute_score(pts)
        return -score

    def objective_full(points_flat):
        """
        Objective function for full 26D optimization.
        """
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        return -score

    def random_walk_polish(pts, max_iter=50, base_step=0.05, rng=None):
        """
        Simple stochastic hill climbing.
        """
        if rng is None:
            rng = np.random.RandomState()

        current_score = compute_score(pts)
        current_pts = pts.copy()
        step = base_step

        for _ in range(max_iter):
            if time.time() > start_time + time_limit - 2:
                break

            i = rng.randint(N)
            angle = rng.uniform(0, 2 * np.pi)
            new_pt = current_pts[i].copy()
            new_pt[0] += step * np.cos(angle)
            new_pt[1] += step * np.sin(angle)

            new_pts = current_pts.copy()
            new_pts[i] = new_pt

            new_score = compute_score(new_pts)

            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                step *= 1.02
            else:
                step *= 0.95
                if step < 1e-5:
                    step = 1e-5
        return current_pts, current_score

    def coordinate_descent_polish(pts, max_iter=20, step=0.01):
        """
        Simple coordinate descent to fine-tune.
        """
        current_pts = pts.copy()
        current_score = compute_score(pts)

        for _ in range(max_iter):
            if time.time() > start_time + time_limit - 1:
                break
            improved = False
            for i in range(N):
                for d in range(2):
                    # Try small steps in + and - direction
                    for delta in [step, -step, step*0.5, -step*0.5, step*2, -step*2]:
                        trial = current_pts.copy()
                        trial[i, d] += delta
                        trial_score = compute_score(trial)
                        if trial_score > current_score:
                            current_pts = trial
                            current_score = trial_score
                            improved = True
                            break
            if not improved:
                step *= 0.8
                if step < 1e-6:
                    break
        return current_pts, current_score

    def fourier_points(coeffs):
        """
        coeffs: array of length 2*K+1 (a0, a1, b1, a2, b2, ...)
        r(theta) = a0 + sum_k (a_k cos(k theta) + b_k sin(k theta))
        Place 13 points at angles 2*pi*k/13.
        """
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.full(N, coeffs[0])
        for k in range(1, len(coeffs)//2 + 1):
            ak = coeffs[2*k - 1]
            bk = coeffs[2*k]
            radii += ak * np.cos(k * angles) + bk * np.sin(k * angles)
        # Ensure positive radii
        radii = np.maximum(radii, 0.1)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        return np.column_stack([xs, ys])

    def objective_fourier(coeffs):
        pts = fourier_points(coeffs)
        score = compute_score(pts)
        return -score

    best_score = -float('inf')
    best_points = None

    def try_improve(pts):
        nonlocal best_score, best_points
        if time.time() > start_time + time_limit - 5:
            return

        current_pts = pts.copy()

        # 1. Random Walk
        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        rw_pts, rw_score = random_walk_polish(current_pts, max_iter=30, base_step=0.05, rng=rng)
        if rw_score > best_score:
            best_score = rw_score
            best_points = rw_pts.copy()
        current_pts = rw_pts

        # 2. Nelder-Mead
        if time.time() > start_time + time_limit - 10:
            return
        try:
            res = opt.minimize(
                objective_full,
                current_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            nm_pts = res.x.reshape(-1, 2)
            nm_score = compute_score(nm_pts)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_pts.copy()
            current_pts = nm_pts
        except Exception:
            pass

        # 3. Coordinate Descent / Fine Polish
        if time.time() > start_time + time_limit - 2:
            return
        cd_pts, cd_score = coordinate_descent_polish(current_pts, max_iter=10, step=0.01)
        if cd_score > best_score:
            best_score = cd_score
            best_points = cd_pts.copy()

    # Phase 1: Symmetric Subspace Optimization (High quality baseline)
    initializations_sym = [
        np.ones(N),
        np.ones(N) + 0.05 * np.sin(2 * np.pi * np.arange(N) / N),
        np.ones(N) + 0.05 * np.sin(4 * np.pi * np.arange(N) / N),
    ]

    for init_rad in initializations_sym:
        if time.time() > start_time + time_limit - 20:
            break
        try:
            res = opt.minimize(
                objective_symmetric,
                init_rad,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            sym_pts = points_from_radii(res.x)
            try_improve(sym_pts)
        except Exception:
            continue

    # Phase 2: Fourier-based convex shape optimization
    # Use a low-dimensional Fourier representation to explore convex shapes
    # coeffs: [a0, a1, b1, a2, b2, a3, b3] -> 7 parameters
    fourier_inits = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]),
        np.array([1.0, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01]),
    ]

    for init_coeffs in fourier_inits:
        if time.time() > start_time + time_limit - 15:
            break
        try:
            res = opt.minimize(
                objective_fourier,
                init_coeffs,
                method='Powell',
                options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            fourier_pts = fourier_points(res.x)
            try_improve(fourier_pts)
        except Exception:
            continue

    # Phase 3: Diverse Topology Search
    rng_global = np.random.RandomState(42)

    # Candidate 1: Regular Polygon
    try_improve(_regular_polygon(N, 1.0))

    # Candidate 2: 12 Outer + 1 Center
    if time.time() < start_time + time_limit - 20:
        pts_12_1 = np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])])
        try_improve(pts_12_1)

    # Candidate 3: Concentric Circles (7+6, 6+7, 8+5, 5+8)
    for n1 in [6, 7, 8]:
        n2 = N - n1
        for r1 in [0.5, 0.7, 0.85]:
            if time.time() > start_time + time_limit - 10:
                break
            outer = _regular_polygon(n1, 1.0)
            inner = _regular_polygon(n2, r1)
            # Randomly rotate inner
            angle_offset = rng_global.uniform(0, 2*np.pi)
            rot_mat = np.array([[np.cos(angle_offset), -np.sin(angle_offset)],
                                [np.sin(angle_offset), np.cos(angle_offset)]])
            inner_rot = inner @ rot_mat.T

            pts = np.vstack([outer, inner_rot])
            try_improve(pts)

    # Candidate 4: Random Points
    for _ in range(10):
        if time.time() > start_time + time_limit - 5:
            break
        # Random points in disk
        r = np.sqrt(rng_global.uniform(0, 1, N))
        theta = rng_global.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        try_improve(pts)

    if best_points is None:
        best_points = _regular_polygon(N, 1.0)

    return best_points
