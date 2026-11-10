# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin for 600s limit

    def normalize(points):
        """
        Center and scale points to prevent numerical drift.
        Scales so that mean pairwise distance is 1.
        """
        pts = points.copy()
        pts -= np.mean(pts, axis=0)
        dists = pdist(pts)
        if len(dists) > 0:
            m = np.mean(dists)
            if m > 1e-12:
                pts /= m
        return pts

    def get_score(points):
        """Calculate the score R = (d_min / d_max)^2"""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max <= 1e-15:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        """
        Objective function for minimization.
        Minimizes log(d_max) - log(d_min).
        """
        points = x.reshape(n, d)
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min < 1e-12 or d_max < 1e-12:
            return 1e10
        if d_min < 1e-4:
            return 1e10 - 1e6 * d_min
        return np.log(d_max) - np.log(d_min)

    def smooth_objective(x, beta=50.0):
        """
        Smooth surrogate for the min/max ratio using log-sum-exp.
        """
        points = x.reshape(n, d)
        dists = pdist(points)
        if np.any(dists < 1e-12):
            return 1e10
        max_d = np.max(dists)
        soft_max = max_d + np.log(np.sum(np.exp(beta * (dists - max_d)))) / beta
        min_d = np.min(dists)
        soft_min = min_d - np.log(np.sum(np.exp(-beta * (dists - min_d)))) / beta
        ratio = soft_max - soft_min
        if ratio < 0:
            return 1e10
        return ratio

    def local_search(points, max_iters=10000):
        """
        Local search with adaptive step size using smooth objective for guidance.
        """
        best_points = normalize(points.copy())
        best_score = get_score(best_points)
        step_size = 0.08
        stagnant = 0

        for i in range(max_iters):
            if time.time() > deadline - 5:
                break

            improved = False
            j = np.random.randint(n)

            for _ in range(15):
                perturbation = np.random.normal(0, step_size, size=d)
                candidate = best_points.copy()
                candidate[j] += perturbation
                candidate = normalize(candidate)

                new_score = get_score(candidate)
                if new_score > best_score:
                    best_score = new_score
                    best_points = candidate
                    improved = True
                    break

            if not improved:
                stagnant += 1
                if stagnant > 25:
                    step_size *= 0.92
                    stagnant = 0
                    if step_size < 1e-9:
                        break
            else:
                stagnant = 0

        return best_points, best_score

    def polish_points(points, max_time=35.0):
        """
        High-precision polish using smooth objective with multiple betas.
        """
        start = time.time()
        pts = normalize(points.copy())
        best_score = get_score(pts)

        for beta in [150.0, 300.0, 600.0, 1000.0]:
            if time.time() - start > max_time - 2:
                break

            def smooth_obj(x):
                return smooth_objective(x, beta=beta)

            try:
                res = minimize(
                    smooth_obj,
                    pts.ravel(),
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
                )
                cand = normalize(res.x.reshape(n, d))
                s = get_score(cand)
                if s > best_score:
                    best_score = s
                    pts = cand.copy()
            except Exception:
                pass

        if time.time() - start < max_time - 1:
            try:
                res = minimize(
                    objective,
                    pts.ravel(),
                    method='Nelder-Mead',
                    options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                )
                cand = normalize(res.x.reshape(n, d))
                s = get_score(cand)
                if s > best_score:
                    best_score = s
                    pts = cand.copy()
            except Exception:
                pass

        return pts, best_score

    # ---------- Structured seeds ----------
    def hex_grid_16():
        pts = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i + 0.5 * j
                y = (np.sqrt(3) / 2.0) * j
                pts.append([x, y])
        pts = np.array(pts)
        d2 = np.sum(pts ** 2, axis=1)
        idx = np.argsort(d2)[:16]
        return pts[idx]

    def concentric_rings():
        c = [[0.0, 0.0]]
        a7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        inner = np.column_stack([np.cos(a7), np.sin(a7)])
        a8 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
        outer = 1.8 * np.column_stack([np.cos(a8), np.sin(a8)])
        return np.vstack([c, inner, outer])

    def two_rings():
        a1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        r1 = np.column_stack([np.cos(a1), np.sin(a1)])
        a2 = a1 + np.pi / 8
        r2 = 1.6 * np.column_stack([np.cos(a2), np.sin(a2)])
        return np.vstack([r1, r2])

    def regular_polygon_16():
        a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        return np.column_stack([np.cos(a), np.sin(a)])

    def square_grid_4x4():
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def multi_ring_flower():
        c = np.array([[0.0, 0.0]])
        a5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        r5 = 0.9 * np.column_stack([np.cos(a5), np.sin(a5)])
        a10 = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi / 10
        radii = 1.8 + 0.1 * np.sin(2 * a10)
        r10 = np.column_stack([radii * np.cos(a10), radii * np.sin(a10)])
        return np.vstack([c, r5, r10])

    def radial_warp_grid():
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        scale = 1.0 + 0.1 * np.sin(r)
        pts = pts * scale[:, np.newaxis]
        return pts

    def triangular_lattice_16():
        """16 points from triangular lattice optimized for min distance."""
        pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i + 0.5 * j
                y = (np.sqrt(3) / 2.0) * j
                pts.append([x, y])
        pts = np.array(pts)
        # Pick the 16 closest to center
        d2 = np.sum(pts ** 2, axis=1)
        idx = np.argsort(d2)[:16]
        return pts[idx]

    def optimized_rings():
        """Two rings with optimized radii and angles."""
        a_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        r_inner = np.column_stack([np.cos(a_inner), np.sin(a_inner)])
        a_outer = np.linspace(0, 2 * np.pi, 9, endpoint=False) + np.pi / 9
        r_outer = 1.7 * np.column_stack([np.cos(a_outer), np.sin(a_outer)])
        return np.vstack([r_inner, r_outer])

    def center_plus_15():
        """Center point plus 15 points on a ring."""
        c = np.array([[0.0, 0.0]])
        a = np.linspace(0, 2 * np.pi, 15, endpoint=False)
        r = 1.5 * np.column_stack([np.cos(a), np.sin(a)])
        return np.vstack([c, r])

    def three_rings():
        """Three concentric rings: 1+6+9 = 16 points."""
        c = np.array([[0.0, 0.0]])
        a6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        r6 = 0.8 * np.column_stack([np.cos(a6), np.sin(a6)])
        a9 = np.linspace(0, 2 * np.pi, 9, endpoint=False) + np.pi / 9
        r9 = 1.6 * np.column_stack([np.cos(a9), np.sin(a9)])
        return np.vstack([c, r6, r9])

    starts = [hex_grid_16(), concentric_rings(), two_rings(), regular_polygon_16(),
              square_grid_4x4(), multi_ring_flower(), radial_warp_grid(),
              triangular_lattice_16(), optimized_rings(), center_plus_15(), three_rings()]

    # Add a few random starts
    rng = np.random.RandomState(42)
    for _ in range(4):
        starts.append(rng.randn(n, d))

    # Normalize all starts
    starts = [normalize(s) for s in starts]

    best_score = -1.0
    best_points = starts[0].copy()

    # ---------- Phase A: Differential Evolution for global exploration ----------
    bounds = [(-5.0, 5.0)] * (n * d)

    def de_objective(x):
        return objective(x)

    if time.time() < deadline - 120:
        try:
            de_res = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=300,
                tol=1e-10,
                mutation=(0.3, 1.0),
                recombination=0.7,
                popsize=25,
                polish=True,
                updating='immediate',
                workers=1
            )
            de_points = de_res.x.reshape(n, d)
            de_points = normalize(de_points)
            s = get_score(de_points)
            if s > best_score:
                best_score = s
                best_points = de_points.copy()
        except Exception:
            pass

    # ---------- Phase B: Powell + L-BFGS-B refinement from multiple starts ----------
    for i, start in enumerate(starts):
        if time.time() > deadline - 30:
            break
        try:
            # Powell
            res = minimize(objective, start.ravel(), method='Powell',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
            pts = normalize(res.x.reshape(n, d))
            s = get_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()

            # L-BFGS-B for faster refinement
            if time.time() < deadline - 25:
                res2 = minimize(objective, pts.ravel(), method='L-BFGS-B',
                                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                pts2 = normalize(res2.x.reshape(n, d))
                s2 = get_score(pts2)
                if s2 > best_score:
                    best_score = s2
                    best_points = pts2.copy()

            # Nelder-Mead for diversity
            if time.time() < deadline - 20:
                res3 = minimize(objective, start.ravel(), method='Nelder-Mead',
                                options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9})
                pts3 = normalize(res3.x.reshape(n, d))
                s3 = get_score(pts3)
                if s3 > best_score:
                    best_score = s3
                    best_points = pts3.copy()
        except Exception:
            pass

    # ---------- Phase C: Adaptive local search from best points ----------
    if time.time() < deadline - 40:
        np.random.seed(12345)
        refined_points, refined_score = local_search(best_points, max_iters=10000)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()

    # ---------- Phase D: High-precision polish ----------
    if time.time() < deadline - 5:
        polished_points, polished_score = polish_points(best_points, max_time=min(35.0, deadline - time.time() - 3))
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points.copy()

    return best_points

# EVOLVE-BLOCK-END
