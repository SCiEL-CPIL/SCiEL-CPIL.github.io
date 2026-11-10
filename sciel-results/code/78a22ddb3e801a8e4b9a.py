# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
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
    deadline = start_time + 590.0

    def score(p):
        """Compute R = (d_min/d_max)^2, invariant to translation and uniform scaling."""
        dists = pdist(p)
        if len(dists) == 0:
            return 0.0
        dmin = np.min(dists)
        dmax = np.max(dists)
        if dmax <= 1e-15:
            return 0.0
        r = dmin / dmax
        return r * r

    def objective(x):
        """Minimize -log(dmin/dmax) = log(dmax) - log(dmin), with heavy penalty for near-degenerate configs."""
        p = x.reshape(n, d)
        dists = pdist(p)
        dmin = np.min(dists)
        dmax = np.max(dists)
        if dmin <= 1e-10 or dmax <= 1e-10:
            return 1e8
        return np.log(dmax) - np.log(dmin)

    def normalize(p):
        """Center at origin and scale so that mean pairwise distance = 1."""
        p = p - p.mean(axis=0)
        dists = pdist(p)
        m = np.mean(dists)
        if m > 1e-15:
            p /= m
        return p

    # ---------- Known good starting configurations ----------
    def hex_grid_16():
        """16 points from a hexagonal lattice, centered."""
        pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i + 0.5 * j
                y = (np.sqrt(3) / 2.0) * j
                pts.append([x, y])
        pts = np.array(pts)
        # Keep the 16 closest to origin
        d2 = np.sum(pts ** 2, axis=1)
        idx = np.argsort(d2)[:16]
        return pts[idx]

    def concentric_rings():
        """1 center + 7 inner ring + 8 outer ring."""
        c = [[0.0, 0.0]]
        a7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        inner = np.column_stack([np.cos(a7), np.sin(a7)])
        a8 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
        outer = 1.8 * np.column_stack([np.cos(a8), np.sin(a8)])
        return np.vstack([c, inner, outer])

    def two_rings():
        """Two rings of 8 points each."""
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

    starts = [hex_grid_16(), concentric_rings(), two_rings(), regular_polygon_16(), square_grid_4x4()]
    # Add a few random starts
    rng = np.random.RandomState(42)
    for _ in range(4):
        starts.append(rng.randn(n, d))

    # Normalize all starts
    starts = [normalize(s) for s in starts]

    best_score = -1.0
    best_points = starts[0].copy()

    # ---------- Phase A: Differential Evolution for global exploration ----------
    # Bounds: after normalization the points should be within [-5, 5]
    bounds = [(-5.0, 5.0)] * (n * d)

    def de_objective(x):
        """Wrap objective for DE; add small penalty if points go far out of range."""
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
                popsize=20,
                polish=True,
                updating='immediate',
                workers=1
            )
            de_points = de_res.x.reshape(n, d)
            de_points = normalize(de_points)
            s = score(de_points)
            if s > best_score:
                best_score = s
                best_points = de_points.copy()
            print(f"[DE] score={s:.6f}, time={time.time()-start_time:.1f}s")
        except Exception as e:
            print(f"[DE] exception: {e}")

    # ---------- Phase B: Powell refinement from multiple starts ----------
    for i, start in enumerate(starts):
        if time.time() > deadline - 30:
            break
        try:
            # Powell
            res = minimize(objective, start.ravel(), method='Powell',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
            pts = normalize(res.x.reshape(n, d))
            s = score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()

            # Also try Nelder-Mead from the same start for diversity
            if time.time() < deadline - 25:
                res2 = minimize(objective, start.ravel(), method='Nelder-Mead',
                                options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9})
                pts2 = normalize(res2.x.reshape(n, d))
                s2 = score(pts2)
                if s2 > best_score:
                    best_score = s2
                    best_points = pts2.copy()
        except Exception as e:
            print(f"[Powell {i}] exception: {e}")

    # ---------- Phase C: Adaptive local search from best points ----------
    def local_search(points, max_iters=8000):
        """Adaptive-step random perturbation local search."""
        bp = points.copy()
        bs = score(bp)
        step = 0.05
        stagnant = 0

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            for j in range(n):
                for _ in range(3):
                    pert = np.random.randn(d) * step
                    cand = bp.copy()
                    cand[j] += pert
                    cand = normalize(cand)
                    ns = score(cand)
                    if ns > bs + 1e-12:
                        bs = ns
                        bp = cand
                        improved = True
            if improved:
                stagnant = 0
            else:
                stagnant += 1
                if stagnant > 8:
                    step *= 0.75
                    stagnant = 0
                    if step < 1e-8:
                        break
        return bp, bs

    if time.time() < deadline - 15:
        np.random.seed(777)
        refined, rs = local_search(best_points, max_iters=6000)
        if rs > best_score:
            best_score = rs
            best_points = refined

    # ---------- Phase D: One more Powell pass from the best ----------
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_points.ravel(), method='Powell',
                           options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-11})
            pts = normalize(res.x.reshape(n, d))
            s = score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # Final sanity: ensure finite, shape (16,2)
    if not np.all(np.isfinite(best_points)):
        best_points = starts[0].copy()
    return best_points

# EVOLVE-BLOCK-END
