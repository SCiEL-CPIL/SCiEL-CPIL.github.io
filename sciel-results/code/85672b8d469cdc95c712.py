"""Pack 50 equal circles in unit circle using soft-min gradient optimization."""
import numpy as np
import time

N = 50
DEADLINE = 130.0  # seconds, safety margin before 150s kill


def _compute_score(pts):
    """Compute the valid packing radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary constraint
    r_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    if np.any(r_boundary < -1e-9):
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    r = min(r_pair, np.min(r_boundary))
    if r < 0:
        return 0.0
    return float(r)


def _soft_min_obj(pts, beta):
    """
    Smooth surrogate for -min(min_pairwise_dist/2, min_boundary).
    Returns (value, gradient).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Boundary distances: d_i = 1 - |p_i|
    norms = np.linalg.norm(pts, axis=1)
    d_boundary = 1.0 - norms
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2)
    # Exclude self
    np.fill_diagonal(dist, np.inf)
    
    # We want to maximize min of all distances (boundary and pairwise/2)
    # Let's work with the raw distances and take min
    # Boundary contributions: d_boundary[i]
    # Pairwise contributions: dist[i,j] for i<j
    
    # Collect all relevant distances
    # For efficiency, work with log-sum-exp
    # We want softmin of {d_boundary[i], dist[i,j]/2 for i<j}
    
    # Actually, let's define the objective as:
    # maximize r such that all constraints hold
    # soft surrogate: -softmin(all_distances)
    # where all_distances includes boundary dists and half pairwise dists
    
    # Build array of all distances
    # Boundary: N values
    # Pairwise: N*(N-1)/2 values
    # Total: N + N*(N-1)/2 = 50 + 1225 = 1275
    
    # For log-sum-exp stability
    # softmin(x) = -logsumexp(-x)
    
    # Let's compute -x for all distances
    neg_boundary = -d_boundary  # shape (N,)
    # For pairwise, we only need upper triangle
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]  # shape (N*(N-1)/2,)
    neg_pairs = -pair_dists / 2.0  # we want min of dist/2
    
    # Combine all
    all_neg = np.concatenate([neg_boundary, neg_pairs])
    
    # logsumexp for numerical stability
    m = np.max(all_neg)
    if m > 500:
        # overflow protection
        logsumexp_val = m + np.log(np.sum(np.exp(all_neg - m)))
    else:
        logsumexp_val = np.log(np.sum(np.exp(all_neg)))
    
    softmin_val = -logsumexp_val
    # Objective: maximize softmin, so minimize -softmin
    obj_val = -softmin_val
    
    # Gradient
    # d/dx softmin(x) = - sum_j exp(-x_j) * (-dx_j/dx) / sum_j exp(-x_j)
    # = sum_j exp(-x_j) * dx_j/dx / sum_j exp(-x_j)
    # where x_j are the distances
    
    # Let's compute weights
    exp_neg = np.exp(all_neg - m)  # shifted for stability
    sum_exp = np.sum(exp_neg)
    weights = exp_neg / sum_exp  # shape (1275,)
    
    # Gradient of obj = -softmin, so grad_obj = -grad_softmin
    # grad_softmin[i] = sum_j weights_j * d(x_j)/d(pts[i])
    
    grad = np.zeros_like(pts)
    
    # Boundary contribution: x_j = 1 - |p_j| for j in 0..N-1
    # d(x_j)/d(p_j) = -p_j / |p_j|
    w_boundary = weights[:N]  # shape (N,)
    # grad_boundary[i] = w_boundary[i] * (-pts[i] / norms[i])
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    grad += w_boundary[:, np.newaxis] * (-pts / safe_norms[:, np.newaxis])
    
    # Pairwise contribution: x_k = dist(i,k)/2 for pair (i,k), i<k
    # d(x_k)/d(p_i) = (p_i - p_k) / dist(i,k)
    # d(x_k)/d(p_k) = (p_k - p_i) / dist(i,k)
    
    w_pairs = weights[N:]  # shape (1225,)
    ii, jj = iu  # indices
    # For each pair (ii[m], jj[m]), weight w_pairs[m]
    # Contribution to grad[ii[m]]: w_pairs[m] * (pts[ii[m]] - pts[jj[m]]) / dist[ii[m], jj[m]]
    # Contribution to grad[jj[m]]: w_pairs[m] * (pts[jj[m]] - pts[ii[m]]) / dist[ii[m], jj[m]]
    
    # Vectorized:
    diff_pairs = pts[ii] - pts[jj]  # shape (1225, 2)
    dist_pairs = pair_dists  # shape (1225,)
    safe_dists = np.where(dist_pairs > 1e-12, dist_pairs, 1.0)
    unit_diff = diff_pairs / safe_dists[:, np.newaxis]  # shape (1225, 2)
    
    # grad[ii] += sum over m where ii[m]=i of w_pairs[m] * unit_diff[m]
    # grad[jj] -= sum over m where jj[m]=j of w_pairs[m] * unit_diff[m]
    
    # Use np.add.at for accumulation
    np.add.at(grad, ii, w_pairs[:, np.newaxis] * unit_diff)
    np.add.at(grad, jj, -w_pairs[:, np.newaxis] * unit_diff)
    
    # Objective gradient is -softmin gradient
    grad_obj = -grad
    
    return obj_val, grad_obj


def _hexagonal_init():
    """Generate initial points using hexagonal lattice, pick best N."""
    # Try several spacings and pick the one that gives best initial score
    best_pts = None
    best_score = 0.0
    for spacing in np.linspace(0.28, 0.38, 11):
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Pick the N closest to center for better initial packing
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            sc = _compute_score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
    if best_pts is not None:
        return best_pts
    # Fallback: uniform random
    return np.random.randn(N, 2) * 0.5


def _ring_init():
    """Initialize with concentric ring structure (1 + 6 + 12 + 15 + 16 = 50)."""
    # Known good structure: center + 4 rings
    # Ring radii roughly: 0, r1, r2, r3, r4
    # With 50 circles, try rings of sizes 1, 6, 12, 15, 16
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Ring 1: 6 points
    r1 = 0.25
    for i in range(6):
        ang = 2 * np.pi * i / 6
        pts.append([r1 * np.cos(ang), r1 * np.sin(ang)])
    # Ring 2: 12 points
    r2 = 0.48
    for i in range(12):
        ang = 2 * np.pi * i / 12 + np.pi / 12
        pts.append([r2 * np.cos(ang), r2 * np.sin(ang)])
    # Ring 3: 15 points
    r3 = 0.68
    for i in range(15):
        ang = 2 * np.pi * i / 15
        pts.append([r3 * np.cos(ang), r3 * np.sin(ang)])
    # Ring 4: 16 points
    r4 = 0.85
    for i in range(16):
        ang = 2 * np.pi * i / 16 + np.pi / 16
        pts.append([r4 * np.cos(ang), r4 * np.sin(ang)])
    
    pts = np.array(pts)
    return pts


def construct():
    start_time = time.time()
    
    # Try multiple initializations
    inits = []
    inits.append(_ring_init())
    inits.append(_hexagonal_init())
    
    # Add a few random perturbations of the ring init
    rng = np.random.default_rng(42)
    for _ in range(3):
        perturbed = inits[0].copy()
        perturbed += rng.normal(0, 0.02, size=perturbed.shape)
        inits.append(perturbed)
    
    best_pts = None
    best_score = 0.0
    
    for init_pts in inits:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Optimize using L-BFGS-B
        # We need to handle boundary constraints. 
        # Strategy: use a penalty for points outside, and let the optimizer handle it.
        # Alternatively, transform coordinates. For simplicity, use penalty.
        
        def objective_with_penalty(pts_flat):
            pts = pts_flat.reshape(-1, 2)
            val, grad = _soft_min_obj(pts, beta=50.0)
            # Penalty for boundary violation
            norms = np.linalg.norm(pts, axis=1)
            viol = np.maximum(0, norms - 1.0)
            penalty = 1000.0 * np.sum(viol ** 2)
            # Penalty gradient
            grad_pen = np.zeros_like(pts)
            mask = norms > 1.0
            if np.any(mask):
                safe_norms = np.where(norms > 1e-12, norms, 1.0)
                grad_pen[mask] = 2000.0 * viol[mask] * pts[mask] / safe_norms[mask, np.newaxis]
            return val + penalty, (grad + grad_pen).flatten()
        
        x0 = init_pts.flatten()
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_with_penalty,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            opt_pts = result.x.reshape(-1, 2)
        except Exception:
            opt_pts = init_pts
        
        # Compute score
        sc = _compute_score(opt_pts)
        if sc > best_score:
            best_score = sc
            best_pts = opt_pts.copy()
        
        # Also check the initial point
        sc_init = _compute_score(init_pts)
        if sc_init > best_score:
            best_score = sc_init
            best_pts = init_pts.copy()
    
    # Final polish: run more iterations on the best result
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        def objective_with_penalty(pts_flat):
            pts = pts_flat.reshape(-1, 2)
            val, grad = _soft_min_obj(pts, beta=100.0)
            norms = np.linalg.norm(pts, axis=1)
            viol = np.maximum(0, norms - 1.0)
            penalty = 1000.0 * np.sum(viol ** 2)
            grad_pen = np.zeros_like(pts)
            mask = norms > 1.0
            if np.any(mask):
                safe_norms = np.where(norms > 1e-12, norms, 1.0)
                grad_pen[mask] = 2000.0 * viol[mask] * pts[mask] / safe_norms[mask, np.newaxis]
            return val + penalty, (grad + grad_pen).flatten()
        
        x0 = best_pts.flatten()
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_with_penalty,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            opt_pts = result.x.reshape(-1, 2)
            sc = _compute_score(opt_pts)
            if sc > best_score:
                best_score = sc
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback
        best_pts = _hexagonal_init()
    
    return best_pts
