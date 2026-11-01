"""Optimize 50 circle centres in the unit disk via soft-min + L-BFGS-B with high-precision polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 130.0  # Safety margin before 150s limit


def _score(points):
    """Compute the packing radius (higher is better)."""
    if points.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists_to_origin = np.sqrt(np.sum(points ** 2, axis=1))
    boundary_margin = 1.0 - np.max(dists_to_origin)
    if boundary_margin <= 0:
        return 0.0
    d = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    if min_pair <= 0:
        return 0.0
    return min(min_pair / 2.0, boundary_margin)


def _soft_min_obj_and_grad(x, beta):
    """
    Soft-min objective using log-sum-exp for smoothness.
    Minimizes -r where r is the packing radius, using a smooth approximation.
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Distances to boundary (1 - |p_i|)
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary_dists = 1.0 - radii
    
    # Combine all constraints into a single vector of "margins"
    # We want to maximize the minimum margin
    # Margin for pair (i,j): dists[i,j] / 2
    # Margin for boundary: boundary_dists[i]
    pair_margins = dists / 2.0
    # Flatten pair margins (upper triangle only to avoid duplicates)
    iu = np.triu_indices(N, k=1)
    pair_margin_vec = pair_margins[iu]
    
    # All margins
    all_margins = np.concatenate([pair_margin_vec, boundary_dists])
    
    # Soft-min via log-sum-exp: -log(sum(exp(-beta * m))) / beta
    # This approximates min(m) as beta -> inf
    neg_beta_m = -beta * all_margins
    max_val = np.max(neg_beta_m)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_m - max_val)))
    soft_min_val = -log_sum_exp / beta
    
    # Objective: minimize -soft_min_val (i.e., maximize soft_min_val)
    obj = -soft_min_val
    
    # Gradient with respect to x
    # d(obj)/dx = -d(soft_min_val)/dx
    # soft_min_val = -log(sum(exp(-beta*m_k)))/beta
    # d(soft_min_val)/dm_k = -exp(-beta*m_k) / sum(exp(-beta*m_j))
    # So d(obj)/dm_k = exp(-beta*m_k) / sum(exp(-beta*m_j))
    
    weights = np.exp(neg_beta_m - max_val)
    weights /= np.sum(weights)
    
    # Gradient of each margin with respect to x
    # For pair (i,j): m_ij = |p_i - p_j| / 2
    # d(m_ij)/dx = (p_i - p_j) / (2 * |p_i - p_j|) * (1/2 for i, -1/2 for j)
    # Actually: d(|p_i - p_j|/2)/dp_i = (p_i - p_j) / (2 * |p_i - p_j|)
    #           d(|p_i - p_j|/2)/dp_j = (p_j - p_i) / (2 * |p_i - p_j|)
    
    grad = np.zeros_like(x)
    
    # Pair gradients
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d = dists[i, j]
        if d > 1e-12:
            w = weights[idx]
            diff_ij = pts[i] - pts[j]
            grad[2*i:2*i+2] += w * diff_ij / d
            grad[2*j:2*j+2] -= w * diff_ij / d
    
    # Boundary gradients
    for i in range(N):
        r = radii[i]
        if r > 1e-12:
            w = weights[len(pair_margin_vec) + i]
            grad[2*i:2*i+2] += w * pts[i] / r
    
    return obj, grad


def _hex_lattice_init():
    """Generate hexagonal lattice initial configuration."""
    best_pts = None
    best_score = -1
    for trial in range(60):
        spacing = 2.0 / (np.sqrt(N) + 1.0 + 0.01 * trial)
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x ** 2 + y ** 2 <= (1.0 - spacing / 2.0) ** 2:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= N:
            radii = np.sqrt(np.sum(pts ** 2, axis=1))
            idx = np.argsort(radii, kind="stable")[:N]
            candidate = pts[idx]
            s = _score(candidate)
            if s > best_score:
                best_score = s
                best_pts = candidate
    if best_pts is not None:
        return best_pts
    return _random_init(42)


def _random_init(seed=None):
    """Generate random initial configuration."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((N, 2))
    for i in range(N):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x ** 2 + y ** 2 <= 0.85 ** 2:
                pts[i] = [x, y]
                break
    return pts


def _perturb(pts, sigma, seed=None):
    """Perturb points with Gaussian noise and clamp to disk."""
    rng = np.random.default_rng(seed)
    new_pts = pts + rng.normal(0, sigma, size=pts.shape)
    r = np.sqrt(np.sum(new_pts ** 2, axis=1))
    mask = r > 0.95
    if np.any(mask):
        scale = 0.95 / r[mask]
        new_pts[mask] *= scale[:, None]
    return new_pts


def _optimize_for_radius(x0, beta, maxiter=2000, gtol=1e-8, deadline=None):
    """Run L-BFGS-B optimization for a given beta."""
    def obj_and_grad(x):
        return _soft_min_obj_and_grad(x, beta)
    
    try:
        result = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'gtol': gtol,
                'ftol': 1e-12,
                'maxfun': maxiter * 2
            }
        )
        return result.x
    except Exception:
        return x0


def construct():
    """Construct centres of N equal circles inside the unit circle."""
    start_time = time.time()
    
    # Start with hexagonal lattice
    pts = _hex_lattice_init()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Adaptive beta schedule: start moderate, increase over iterations
    # Phase 1: moderate beta for global-ish search
    # Phase 2: high beta for precise local optimization
    # Phase 3: very high beta for final polish
    
    betas_phase1 = [50, 100, 200, 500]
    betas_phase2 = [1000, 2000, 5000]
    betas_phase3 = [10000, 20000, 50000]
    
    # Phase 1: Global search with moderate beta
    for beta in betas_phase1:
        if time.time() - start_time > DEADLINE - 30:
            break
        x_opt = _optimize_for_radius(best_pts.flatten(), beta, maxiter=1000, gtol=1e-6)
        x_opt = x_opt.reshape(N, 2)
        # Clamp to disk
        r = np.sqrt(np.sum(x_opt ** 2, axis=1))
        mask = r > 0.99
        if np.any(mask):
            x_opt[mask] *= (0.99 / r[mask])[:, None]
        s = _score(x_opt)
        if s > best_score:
            best_score = s
            best_pts = x_opt.copy()
    
    # Phase 2: Local refinement with higher beta
    for beta in betas_phase2:
        if time.time() - start_time > DEADLINE - 15:
            break
        x_opt = _optimize_for_radius(best_pts.flatten(), beta, maxiter=500, gtol=1e-8)
        x_opt = x_opt.reshape(N, 2)
        r = np.sqrt(np.sum(x_opt ** 2, axis=1))
        mask = r > 0.995
        if np.any(mask):
            x_opt[mask] *= (0.995 / r[mask])[:, None]
        s = _score(x_opt)
        if s > best_score:
            best_score = s
            best_pts = x_opt.copy()
    
    # Phase 3: High-precision final polish
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5:
        for beta in betas_phase3:
            if time.time() - start_time > DEADLINE - 2:
                break
            x_opt = _optimize_for_radius(best_pts.flatten(), beta, maxiter=500, gtol=1e-12)
            x_opt = x_opt.reshape(N, 2)
            r = np.sqrt(np.sum(x_opt ** 2, axis=1))
            mask = r > 0.999
            if np.any(mask):
                x_opt[mask] *= (0.999 / r[mask])[:, None]
            s = _score(x_opt)
            if s > best_score:
                best_score = s
                best_pts = x_opt.copy()
    
    # Try a few random restarts if time permits
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 20:
        for trial in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
            x0 = _random_init(seed=100 + trial).flatten()
            for beta in [200, 1000]:
                if time.time() - start_time > DEADLINE - 5:
                    break
                x_opt = _optimize_for_radius(x0, beta, maxiter=500, gtol=1e-8)
                x_opt = x_opt.reshape(N, 2)
                r = np.sqrt(np.sum(x_opt ** 2, axis=1))
                mask = r > 0.995
                if np.any(mask):
                    x_opt[mask] *= (0.995 / r[mask])[:, None]
                s = _score(x_opt)
                if s > best_score:
                    best_score = s
                    best_pts = x_opt.copy()
    
    # Final validation
    final_score = _score(best_pts)
    if final_score <= 0:
        best_pts = _hex_lattice_init()
    
    return best_pts
