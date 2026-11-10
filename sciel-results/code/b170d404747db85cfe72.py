# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Finds 14 points in 3D maximizing (d_min/d_max)^2.
    Uses a multi-stage approach: sphere gradient descent on smooth proxy,
    then Nelder-Mead on the true objective from diverse initializations.
    """
    n = 14
    d = 3
    t0 = time.time()
    deadline = t0 + 575.0  # safety margin before 600s limit

    def remaining():
        return deadline - time.time()

    def score(pts):
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        dmin = np.min(dists)
        dmax = np.max(dists)
        if dmax <= 1e-15 or dmin <= 1e-15:
            return 0.0
        return (dmin / dmax) ** 2

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return pts / norms

    def fib_sphere(n_pts):
        phi = np.pi * (3.0 - np.sqrt(5.0))
        pts = []
        for i in range(n_pts):
            y = 1.0 - 2.0 * i / (n_pts - 1)
            r = np.sqrt(max(0.0, 1.0 - y * y))
            th = phi * i
            pts.append([np.cos(th) * r, y, np.sin(th) * r])
        return np.array(pts)

    def grad_desc_sphere(pts, iters=3000, lr=0.008, mom=0.9, k=4):
        """Gradient descent minimizing sum 1/d_ij^k on sphere."""
        p = pts.copy()
        v = np.zeros_like(p)
        for it in range(iters):
            if remaining() < 5.0:
                break
            # distance matrix
            diff = p[:, None, :] - p[None, :, :]
            d2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(d2, 1e-12)
            # inv_d^k for k=4 means 1/d^4 = 1/d2^2
            if k == 4:
                w = 1.0 / (d2 ** 2)
            elif k == 2:
                w = 1.0 / d2
            else:
                w = 1.0 / (d2 ** (k // 2))
            np.fill_diagonal(w, 0.0)
            # gradient of sum_{j!=i} 1/d_ij^k w.r.t p_i
            # = -k * sum_j (p_i - p_j) / d_ij^(k+2)
            # For k=4: -4 * sum_j (p_i - p_j) / d_ij^6 = -4 * sum_j (p_i-p_j) * (1/d2^3)
            if k == 4:
                w_grad = 1.0 / (d2 ** 3)
                np.fill_diagonal(w_grad, 0.0)
                grad = -4.0 * (p * w_grad.sum(axis=1, keepdims=True) - (w_grad[:, :, None] * p).sum(axis=1))
            elif k == 2:
                w_grad = 1.0 / (d2 ** 2)
                np.fill_diagonal(w_grad, 0.0)
                grad = -2.0 * (p * w_grad.sum(axis=1, keepdims=True) - (w_grad[:, :, None] * p).sum(axis=1))
            else:
                grad = -k * (p * w_grad.sum(axis=1, keepdims=True) - (w_grad[:, :, None] * p).sum(axis=1))

            v = mom * v - lr * grad
            p = p + v
            p = normalize(p)
        return p

    def nm_refine(pts, maxfev=2000):
        """Nelder-Mead on the true objective, constrained to sphere."""
        def obj(x):
            if remaining() < 2.0:
                return 1e6
            pp = x.reshape(n, d)
            pp = normalize(pp)
            s = score(pp)
            return -s if s > 0 else 1e6

        x0 = pts.flatten()
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxfev': maxfev, 'xatol': 1e-7, 'fatol': 1e-7})
            pp = normalize(res.x.reshape(n, d))
            return pp
        except Exception:
            return pts

    def nm_refine_scaled(pts, scale=0.05, maxfev=1500):
        """NM with scaled perturbation to explore neighborhood."""
        def obj(x):
            if remaining() < 2.0:
                return 1e6
            pp = x.reshape(n, d)
            pp = normalize(pp)
            s = score(pp)
            return -s if s > 0 else 1e6

        # Start from a slightly perturbed version
        x0 = (pts + np.random.randn(n, d) * scale).flatten()
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxfev': maxfev, 'xatol': 1e-7, 'fatol': 1e-7})
            pp = normalize(res.x.reshape(n, d))
            return pp
        except Exception:
            return pts

    # --- Generate diverse initializations ---
    inits = []
    inits.append(fib_sphere(n))

    np.random.seed(12345)
    for trial in range(12):
        pts = np.random.randn(n, d)
        inits.append(normalize(pts))

    # Perturbed Fibonacci with varying amounts
    for eps in [0.05, 0.1, 0.2, 0.3]:
        pts = fib_sphere(n) + np.random.randn(n, d) * eps
        inits.append(normalize(pts))

    # Rotated Fibonacci
    from scipy.spatial.transform import Rotation
    fib = fib_sphere(n)
    for _ in range(4):
        R = Rotation.from_quat(np.random.randn(4)).as_matrix()
        inits.append(normalize(fib @ R.T))

    best_score = -1.0
    best_points = inits[0]

    # --- Phase 1: Gradient descent on smooth proxy from each init ---
    gd_results = []
    for i, init in enumerate(inits):
        if remaining() < 60.0:
            break
        for k in [2, 4]:
            for lr in [0.003, 0.008, 0.015]:
                if remaining() < 30.0:
                    break
                pts = grad_desc_sphere(init, iters=2000, lr=lr, mom=0.9, k=k)
                s = score(pts)
                gd_results.append((s, pts))
                if s > best_score:
                    best_score = s
                    best_points = pts

    # Sort by score, keep top candidates
    gd_results.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate (keep at most 10 distinct)
    top = []
    for s, pts in gd_results:
        dup = False
        for _, p2 in top:
            if np.max(np.abs(pts - p2)) < 1e-3:
                dup = True
                break
        if not dup:
            top.append((s, pts))
        if len(top) >= 10:
            break

    # --- Phase 2: Nelder-Mead refinement on top candidates ---
    for s, pts in top:
        if remaining() < 20.0:
            break
        # Direct NM
        refined = nm_refine(pts, maxfev=1500)
        rs = score(refined)
        if rs > best_score:
            best_score = rs
            best_points = refined

        # Scaled NM from slightly perturbed start
        if remaining() > 30.0:
            refined2 = nm_refine_scaled(pts, scale=0.03, maxfev=1000)
            rs2 = score(refined2)
            if rs2 > best_score:
                best_score = rs2
                best_points = refined2

    # --- Phase 3: If time remains, try additional NM restarts from best ---
    if remaining() > 40.0:
        for _ in range(3):
            if remaining() < 25.0:
                break
            perturbed = normalize(best_points + np.random.randn(n, d) * 0.02)
            refined = nm_refine(perturbed, maxfev=1000)
            rs = score(refined)
            if rs > best_score:
                best_score = rs
                best_points = refined

    # Final sanity
    if not np.all(np.isfinite(best_points)):
        best_points = fib_sphere(n)
        best_points = normalize(best_points)

    return best_points

# EVOLVE-BLOCK-END
