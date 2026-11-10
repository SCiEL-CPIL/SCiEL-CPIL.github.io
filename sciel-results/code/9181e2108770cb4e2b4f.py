# EVOLVE-BLOCK-START
import time
import random
import sys

import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 16


def _radial_warp(points, alpha, beta):
    """Radial distortion: r' = r * (1 + alpha*r + beta*r^2)."""
    r = np.sqrt(np.sum(points ** 2, axis=1, keepdims=True))
    factor = 1.0 + alpha * r + beta * r ** 2
    return points * factor


def _projective_warp(points, a, b, c, d, e, f):
    """2D projective (homography) warp of a point set."""
    denom = 1.0 + c * points[:, 0] + f * points[:, 1]
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    x_new = (a * points[:, 0] + b * points[:, 1] + c) / denom
    y_new = (d * points[:, 0] + e * points[:, 1] + f) / denom
    return np.column_stack([x_new, y_new])


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Returns 16 points in 2D (shape (16,2)) maximizing R = (d_min/d_max)^2.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin under the 630s kill

    def elapsed():
        return time.time() - start_time

    def remaining():
        return deadline - time.time()

    def compute_score(points):
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = d.min()
        d_max = d.max()
        if d_max == 0.0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def normalize_points(points):
        d = pdist(points)
        d_max = d.max()
        if d_max > 1e-12:
            return points / d_max
        return points

    def soft_min_value(points, beta):
        """Smooth approximation of the minimum pairwise distance (d_max normalized to 1)."""
        d = pdist(points)
        d_max = d.max()
        if d_max < 1e-12:
            return 0.0
        dn = d / d_max
        vals = np.exp(-beta * dn)
        s = vals.sum()
        if s == 0.0:
            return 0.0
        return -1.0 / beta * np.log(s)

    def soft_min_grad(points, beta):
        """
        Returns (value, grad) where value = soft_min(d)/d_max and grad is d(value)/d(x).
        d = sqrt(sum_k (x_k - x_j)^2).  d(dn_k)/d(x_a) = (x_a - x_{b(k)}) / (d_k * d_max).
        soft_min = -(1/beta) log sum exp(-beta dn)  =>  d(soft_min)/d(x_a) = sum_k p_k * d(dn_k)/d(x_a)
        where p_k = exp(-beta dn_k)/sum.
        """
        d = pdist(points)
        d_max = d.max()
        if d_max < 1e-12:
            return 0.0, np.zeros_like(points)
        dn = d / d_max
        e = np.exp(-beta * dn)
        s = e.sum()
        if s == 0.0:
            return 0.0, np.zeros_like(points)
        p = e / s  # weights over the 120 distances
        val = -1.0 / beta * np.log(s)
        # build flat index -> (i,j)
        ii, jj = np.triu_indices(N, k=1)
        grad = np.zeros(2 * N)
        # contribution from point i: -p * (x_i - x_j)/(d_k d_max)
        # contribution from point j: +p * (x_i - x_j)/(d_k d_max)
        diff = points[ii] - points[jj]  # (120,2)
        inv = p / (dn * d_max)          # (120,)  = p/(d_k d_max)
        grad_i = -(diff * inv[:, None])  # (120,2)
        grad_j = +(diff * inv[:, None])
        np.add.at(grad, ii * 2, grad_i[:, 0])
        np.add.at(grad, ii * 2 + 1, grad_i[:, 1])
        np.add.at(grad, jj * 2, grad_j[:, 0])
        np.add.at(grad, jj * 2 + 1, grad_j[:, 1])
        return val, grad.reshape(N, 2)

    def anneal(points, time_budget, betas, method="L-BFGS-B", maxfun=1500):
        """Anneal the soft-min temperature, improving the point set."""
        points = normalize_points(points.copy())
        best = points.copy()
        best_score = compute_score(best)
        t0 = time.time()
        for beta in betas:
            if time.time() - t0 > time_budget:
                break
            if method == "L-BFGS-B":
                def obj(x):
                    return soft_min_value(x.reshape(N, 2), beta)

                def objg(x):
                    v, g = soft_min_grad(x.reshape(N, 2), beta)
                    return v, g.flatten()

                try:
                    res = minimize(objg, points.flatten(), method="L-BFGS-B",
                                   jac=True,
                                   options={"maxiter": 400, "ftol": 1e-15,
                                            "gtol": 1e-12, "maxfun": maxfun})
                    cand = normalize_points(res.x.reshape(N, 2))
                except Exception:
                    continue
            else:  # Nelder-Mead
                def obj(x):
                    return -soft_min_value(x.reshape(N, 2), beta)

                try:
                    res = minimize(obj, points.flatten(), method="Nelder-Mead",
                                   options={"maxiter": 2000, "xatol": 1e-9,
                                            "fatol": 1e-10})
                    cand = normalize_points(res.x.reshape(N, 2))
                except Exception:
                    continue
            sc = compute_score(cand)
            if sc > best_score:
                best_score = sc
                best = cand.copy()
            points = best.copy()
            if time.time() - t0 > time_budget:
                break
        return best, best_score

    def nm_polish(points, time_budget):
        """Final Nelder-Mead on the exact (non-smooth) objective."""
        def obj(x):
            p = normalize_points(x.reshape(N, 2))
            d = pdist(p)
            dmin, dmax = d.min(), d.max()
            if dmax == 0:
                return 0.0
            return -(dmin / dmax) ** 2

        x0 = normalize_points(points.copy()).flatten()
        try:
            res = minimize(obj, x0, method="Nelder-Mead",
                           options={"maxiter": 8000, "xatol": 1e-11, "fatol": 1e-11})
            cand = normalize_points(res.x.reshape(N, 2))
            sc = compute_score(cand)
            if sc > compute_score(points):
                return cand, sc
        except Exception:
            pass
        return points, compute_score(points)

    # ------------------------------------------------------------------
    # 1. Build a diverse pool of starting configurations
    # ------------------------------------------------------------------
    candidates = []

    # 4x4 square grid (centered) - the primary strong base
    gx = np.array([[-1.5, -0.5, 0.5, 1.5]] * 4).flatten()
    gy = np.array([-1.5, -0.5, 0.5, 1.5] * 4).flatten()
    grid = np.column_stack([gx, gy])
    candidates.append(grid.copy())

    # Radially warped 4x4 grids (pull corners in / push edge-mid out)
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        for beta in [0.0, -0.05, -0.1, -0.15, 0.05, 0.1]:
            w = _radial_warp(grid, alpha, beta)
            w = w - w.mean(axis=0)
            candidates.append(w)

    # Projective warps of the grid
    proj_list = [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.05, 0.0, 0.0, 0.0, 0.05, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.05),
        (0.05, 0.0, 0.0, 0.0, 0.05, 0.05),
        (0.1, 0.0, 0.0, 0.0, 0.1, 0.0),
        (0.0, 0.1, 0.0, 0.1, 0.0, 0.0),
        (0.05, 0.05, 0.0, 0.05, 0.05, 0.0),
        (-0.05, 0.0, 0.0, 0.0, -0.05, 0.0),
        (0.0, 0.0, 0.02, 0.0, 0.0, 0.02),
        (0.0, 0.0, -0.02, 0.0, 0.0, -0.02),
        (0.08, 0.0, 0.01, 0.0, 0.08, 0.01),
        (0.0, 0.08, 0.0, 0.08, 0.0, 0.01),
    ]
    for a, b, c, d, e, f in proj_list:
        w = _projective_warp(grid, a, b, c, d, e, f)
        w = w - w.mean(axis=0)
        candidates.append(w)

    # Triangular lattice cluster
    s = 1.0
    h = s * np.sqrt(3) / 2
    lat = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            lat.append([i * s + (j % 2) * s / 2, j * h])
    lat = np.array(lat)
    cen = lat.mean(axis=0)
    lat = lat[np.argsort(np.sum((lat - cen) ** 2, axis=1))[:N]]
    candidates.append(lat)

    # Regular 16-gon
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    candidates.append(np.column_stack([np.cos(ang), np.sin(ang)]))

    # Concentric circles
    for n_in, n_out, r_in in [(4, 12, 0.4), (6, 10, 0.45), (8, 8, 0.5)]:
        ia = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
        oa = np.linspace(0, 2 * np.pi, n_out, endpoint=False) + np.pi / n_out
        inner = np.column_stack([r_in * np.cos(ia), r_in * np.sin(ia)])
        outer = np.column_stack([np.cos(oa), np.sin(oa)])
        candidates.append(np.vstack([inner, outer]))

    # Fibonacci spiral
    ga = np.pi * (3 - np.sqrt(5))
    sp = []
    for i in range(16):
        r = np.sqrt(i + 1) / 4
        th = i * ga
        sp.append([r * np.cos(th), r * np.sin(th)])
    candidates.append(np.array(sp))

    # Random seeds
    for seed in range(14):
        np.random.seed(seed * 131 + 17)
        rp = np.random.randn(N, 2)
        rp = normalize_points(rp)
        candidates.append(rp)

    # Jittered grid
    for jseed in range(6):
        np.random.seed(5000 + jseed)
        jg = grid + np.random.randn(N, 2) * 0.08
        candidates.append(jg)

    # ------------------------------------------------------------------
    # 2. Rank candidates by initial score, optimize the best first
    # ------------------------------------------------------------------
    scored = sorted(((compute_score(c), c) for c in candidates),
                    key=lambda t: t[0], reverse=True)

    best_points = scored[0][1].copy()
    best_score = scored[0][0]

    # Warm-start: a quick anneal on the top few to raise the bar early
    for sc, cand in scored[:4]:
        if remaining() < 40:
            break
        p, s = anneal(cand, time_budget=8.0,
                      betas=[10, 30, 100, 300, 800, 2000], maxfun=1200)
        if s > best_score:
            best_score = s
            best_points = p.copy()

    # Full optimization pass over all candidates (best first)
    for rank, (sc, cand) in enumerate(scored):
        if remaining() < 30:
            break
        # budget by rank
        if rank < 4:
            budget = 18.0
        elif rank < 12:
            budget = 10.0
        elif rank < 24:
            budget = 6.0
        else:
            budget = 3.5
        # skip candidates already far below the bar (they can't overtake quickly)
        if sc + 0.5 < best_score and rank >= 12:
            budget = min(budget, 2.0)
        p, s = anneal(cand, time_budget=budget,
                      betas=[10, 30, 100, 300, 800, 2000], maxfun=1500)
        if s > best_score:
            best_score = s
            best_points = p.copy()

    # ------------------------------------------------------------------
    # 3. High-precision refinement of the current best
    # ------------------------------------------------------------------
    if remaining() > 60:
        p, s = anneal(best_points, time_budget=min(60.0, remaining() - 40),
                      betas=[300, 800, 2000, 5000, 12000], maxfun=2500)
        if s > best_score:
            best_score = s
            best_points = p.copy()

    # ------------------------------------------------------------------
    # 4. Final exact Nelder-Mead polish
    # ------------------------------------------------------------------
    if remaining() > 25:
        p, s = nm_polish(best_points, time_budget=min(40.0, remaining() - 15))
        if s > best_score:
            best_score = s
            best_points = p.copy()

    if best_points is None:
        best_points = scored[0][1]

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END
