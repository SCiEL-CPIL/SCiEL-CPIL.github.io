# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _score(pts):
    d = pdist(pts)
    if d.size == 0:
        return 0.0
    dmin = d.min()
    dmax = d.max()
    if dmax <= 0 or not np.isfinite(dmin) or not np.isfinite(dmax):
        return 0.0
    return (dmin / dmax) ** 2


def _init_seeds(n, d, rng):
    seeds = []
    # 1) Fibonacci sphere
    phi = (1 + np.sqrt(5)) / 2.0
    pts = np.zeros((n, d))
    for i in range(n):
        y = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(max(0.0, 1 - y * y))
        th = phi * i * 2 * np.pi
        pts[i] = [r * np.cos(th), r * np.sin(th), y]
    seeds.append(pts)
    # 2) random sphere points, several restarts
    for _ in range(12):
        p = rng.standard_normal((n, d))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        seeds.append(p)
    # 3) Additional random sphere points with different seeds for diversity
    for _ in range(10):
        p = rng.standard_normal((n, d))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        seeds.append(p)
    
    # 4) Random points inside a cube, normalized
    for _ in range(5):
        p = rng.uniform(-1, 1, (n, d))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        seeds.append(p)
        
    return seeds


def min_max_dist_dim3_14() -> np.ndarray:
    n, d = 14, 3
    deadline = time.time() + 540.0  # safety margin under 600s
    rng = np.random.default_rng(12345)

    def objective(v):
        pts = v.reshape(n, d)
        dmat = pdist(pts)
        dmin = dmat.min()
        dmax = dmat.max()
        if dmax <= 1e-9:
            return 1e12
        return -(dmin / dmax) ** 2

    def gradient(v):
        pts = v.reshape(n, d)
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1e-12)
        dmin = dist.min()
        dmax = dist.max()
        if dmax <= 1e-9:
            return np.zeros_like(v)
        ratio = dmin / dmax
        # gradient of -(dmin/dmax)^2
        grad_pts = np.zeros_like(pts)
        # contribution from dmin term: -2*(dmin/dmax^2) * grad dmin
        min_mask = dist <= dmin + 1e-9
        gmin = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                if min_mask[i, j]:
                    gmin[i] += diff[i, j] / dist[i, j]
                    gmin[j] -= diff[i, j] / dist[i, j]
        # contribution from dmax term: -2*(dmin^2/dmax^3) * grad dmax
        max_mask = dist >= dmax - 1e-9
        gmax = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                if max_mask[i, j]:
                    gmax[i] += diff[i, j] / dist[i, j]
                    gmax[j] -= diff[i, j] / dist[i, j]
        grad_pts = -2.0 * (dmin / (dmax ** 2)) * gmin - 2.0 * (dmin ** 2 / (dmax ** 3)) * gmax
        return grad_pts.reshape(-1)

    best_pts = None
    best_val = -np.inf
    seeds = _init_seeds(n, d, rng)

    for seed in seeds:
        if time.time() > deadline:
            break
        v0 = seed.reshape(-1).copy()
        try:
            res = minimize(
                objective,
                v0,
                jac=gradient,
                method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
            )
            cand = res.x.reshape(n, d)
            val = -objective(cand.reshape(-1))
            if np.isfinite(val) and val > best_val:
                best_val = val
                best_pts = cand.copy()
        except Exception:
            continue

    # local polish: small repulsion-based refinement on the best
    if best_pts is not None and time.time() < deadline:
        pts = best_pts.copy()
        step = 0.02
        for _ in range(300):
            if time.time() > deadline:
                break
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, 1e-12)
            dmin = dist.min()
            dmax = dist.max()
            if dmax <= 1e-9:
                break
            force = np.zeros_like(pts)
            for i in range(n):
                for j in range(i + 1, n):
                    u = diff[i, j] / dist[i, j]
                    # repel all pairs, stronger for close pairs
                    w = 1.0 / (dist[i, j] ** 2 + 1e-6)
                    if dist[i, j] >= dmax - 1e-9:
                        w *= 0.3  # soften the diameter pairs
                    force[i] += w * u
                    force[j] -= w * u
            pts = pts + step * force
            # re-center
            pts -= pts.mean(axis=0)
            # re-scale to keep diameter ~1 for stability
            dm = pdist(pts).max()
            if dm > 0:
                pts /= dm
            new_val = _score(pts)
            if new_val > best_val:
                best_val = new_val
                best_pts = pts.copy()
            else:
                pts = best_pts.copy()
                step *= 0.95

    if best_pts is None:
        # fallback: random
        np.random.seed(42)
        best_pts = np.random.randn(n, d)

    # ensure finite
    if not np.all(np.isfinite(best_pts)):
        np.random.seed(42)
        best_pts = np.random.randn(n, d)

    return best_pts


# EVOLVE-BLOCK-END
