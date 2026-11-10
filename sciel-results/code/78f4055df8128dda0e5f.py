# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 30.0

    def elapsed():
        return time.time() - start_time

    def check_deadline():
        return elapsed() > (time_limit - safety_margin)

    # ---------- Symmetric seed: icosahedron (12) + (±1,±1,±1)/√3 (2) ----------
    def make_seed():
        phi = (1 + 5**0.5) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        ], dtype=float)
        norms = np.linalg.norm(ico, axis=1, keepdims=True)
        ico /= norms
        p13 = np.array([1.0, 1.0, 1.0])
        p14 = np.array([-1.0, -1.0, -1.0])
        p13 /= np.linalg.norm(p13)
        p14 /= np.linalg.norm(p14)
        return np.vstack([ico, [p13, p14]])

    # ---------- Objective: negative soft-min of pairwise distances ----------
    # f = -logsumexp(-beta * dists) / beta
    # Minimizing f maximizes the soft-min, which approximates d_min.
    def soft_min_obj(flat, beta):
        pts = flat.reshape(n, 3)
        dists = pdist(pts)
        if len(dists) == 0:
            return 1e10
        if np.any(dists < 1e-15):
            return 1e10
        # Numerically stable logsumexp of -beta*dists
        shifted = -beta * dists
        m = np.max(shifted)
        if not np.isfinite(m):
            return 1e10
        log_sum = m + np.log(np.sum(np.exp(shifted - m)))
        return -log_sum / beta

    def soft_min_grad(flat, beta):
        pts = flat.reshape(n, 3)
        dists = pdist(pts)
        if len(dists) == 0 or np.any(dists < 1e-15):
            return np.zeros_like(flat)
        shifted = -beta * dists
        m = np.max(shifted)
        if not np.isfinite(m):
            return np.zeros_like(flat)
        exp_shifted = np.exp(shifted - m)
        s = np.sum(exp_shifted)
        if s < 1e-300:
            return np.zeros_like(flat)
        w = exp_shifted / s  # weights, sum to 1

        # d(flat[i])/d(dist[k]) for each point i
        # dist[k] = ||p_a - p_b||, so d(dist[k])/d(p_a) = (p_a - p_b)/dist[k]
        #           d(dist[k])/d(p_b) = (p_b - p_a)/dist[k]
        # f = -sum_k w_k * dist[k]  (since -logsumexp(-beta*d)/beta ≈ -weighted sum)
        # Actually f = -logsumexp(-beta*dists)/beta
        # df/d(dist[k]) = -(1/beta) * d/d(dist[k]) logsumexp(-beta*dists)
        #                = -(1/beta) * (1/s) * sum_j exp(-beta*d_j) * (-beta) * delta_{jk}
        #                = (1/s) * exp(-beta*d_k)
        #                = w_k
        # So df/d(dist[k]) = w_k
        # df/d(p_a) = sum_k w_k * d(dist[k])/d(p_a) = sum_k w_k * (p_a - p_b)/dist[k]
        # df/d(p_b) = sum_k w_k * (p_b - p_a)/dist[k]

        # Build index mapping: pdist order is (0,1),(0,2),...,(0,n-1),(1,2),...
        idx = 0
        grad = np.zeros((n, 3))
        for a in range(n - 1):
            for b in range(a + 1, n):
                dk = dists[idx]
                if dk < 1e-15:
                    idx += 1
                    continue
                w_k = w[idx]
                diff = (pts[a] - pts[b]) / dk  # unit vector from b to a
                grad[a] += w_k * diff
                grad[b] -= w_k * diff
                idx += 1

        return grad.reshape(-1)

    # ---------- Optimization loop ----------
    best_pts = make_seed()
    best_score = _score(best_pts)

    # L-BFGS-B with analytical gradient, increasing beta for sharper min
    betas = [60.0, 100.0, 150.0, 200.0, 300.0]
    current = best_pts.copy()

    for beta in betas:
        if check_deadline():
            break
        flat = current.flatten()
        try:
            res = minimize(
                soft_min_obj,
                flat,
                jac=soft_min_grad,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False},
            )
            flat_opt = res.x
        except Exception:
            flat_opt = flat

        opt_pts = flat_opt.reshape(n, 3)
        # Normalize to unit sphere (scale-invariant problem)
        norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        opt_pts /= norms

        sc = _score(opt_pts)
        if sc > best_score:
            best_score = sc
            best_pts = opt_pts.copy()
            current = opt_pts.copy()

        # Small perturbation restart to escape shallow local minima
        if not check_deadline():
            rng = np.random.default_rng(int(beta) * 1000)
            perturbed = current + rng.standard_normal(current.shape) * 0.05
            pnorms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            pnorms[pnorms < 1e-15] = 1.0
            perturbed /= pnorms
            sc_p = _score(perturbed)
            if sc_p > best_score:
                best_score = sc_p
                best_pts = perturbed.copy()
                current = perturbed.copy()

    # Final verification
    if not np.all(np.isfinite(best_pts)):
        best_pts = make_seed()
    if best_pts.shape != (n, d):
        best_pts = make_seed()

    return best_pts

# EVOLVE-BLOCK-END
