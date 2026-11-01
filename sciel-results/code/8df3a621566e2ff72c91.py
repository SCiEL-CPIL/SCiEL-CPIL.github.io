"""Improved packing: hex lattice init + L-BFGS-B optimization of soft-min objective."""
import numpy as np
from scipy.optimize import minimize
import time

N = 300


def _hex_lattice_init(n, r_target=0.053):
    """Generate hexagonal lattice points inside unit disk, pick n closest to center."""
    # Spacing for hex lattice: distance between adjacent centers
    # For radius r, spacing = 2*r
    # We want to fit ~n circles, so area ~ pi, density ~ n
    # Try a range of spacings
    best_points = None
    for scale in np.linspace(0.8, 1.2, 9):
        spacing = 2.0 * r_target * scale
        if spacing <= 0:
            continue
        # Generate hex lattice
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        mask = rad <= 1.0 - r_target
        pts = pts[mask]
        rad = rad[mask]
        if len(pts) >= n:
            # Sort by distance to center, take closest n
            idx = np.argsort(rad, kind="stable")[:n]
            candidate = pts[idx]
            if best_points is None or len(candidate) > len(best_points):
                best_points = candidate
    if best_points is None or len(best_points) < n:
        # Fallback: random
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n))
        best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return best_points[:n]


def _compute_score(pts):
    """Compute the packing score: min of all pairwise distances/2 and (1 - |c_i|)."""
    n = len(pts)
    # Boundary constraint
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary_min = 1.0 - radii
    # Pairwise distances
    # Use chunked computation to save memory
    min_pair = np.inf
    for i in range(n):
        diff = pts[i + 1:] - pts[i]
        d = np.sqrt(np.sum(diff ** 2, axis=1))
        min_pair = min(min_pair, np.min(d))
    return min(min_pair / 2.0, np.min(boundary_min))


def _soft_min_obj(pts, beta=50.0):
    """
    Objective to minimize: -soft_min(pairwise_dists/2, boundary_dists)
    Using log-sum-exp for numerical stability.
    """
    n = len(pts)
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary = 1.0 - radii  # shape (n,)
    
    # Pairwise distances / 2
    # Compute all pairwise distances efficiently
    # Use broadcasting for moderate n
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    # Set diagonal to inf
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    d = d / 2.0  # half-distance
    
    # Combine all constraints into one array
    all_vals = np.concatenate([d.flatten(), boundary])
    
    # Soft min via log-sum-exp: min(x) ≈ -log(sum(exp(-beta*x)))/beta
    # For numerical stability:
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = m - log_sum / beta
    
    return -soft_min_val


def _soft_min_grad(pts, beta=50.0):
    """Gradient of the soft-min objective."""
    n = len(pts)
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary = 1.0 - radii
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    d_half = d / 2.0
    
    all_vals = np.concatenate([d_half.flatten(), boundary])
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax-like weights
    
    # Gradient of soft_min w.r.t. pts
    # d(soft_min)/d(pts_i) = -sum_j w_j * d(constraint_j)/d(pts_i)
    # For pairwise: d(d_ij/2)/d(pts_i) = (pts_i - pts_j) / (2 * d_ij) ... let's be careful
    
    grad = np.zeros_like(pts)
    
    # Boundary gradient: d(boundary_i)/d(pts_i) = -pts_i / |pts_i|
    for i in range(n):
        r = radii[i]
        if r > 1e-12:
            grad[i] += weights[n + i] * (-pts[i] / r)
    
    # Pairwise gradient
    # weights[:n*(n-1)] corresponds to flattened d_half
    # d_half is upper triangle (since diagonal is inf, but actually we have full matrix with diag=inf)
    # Let's just iterate over pairs
    w_pair = weights[:n * n].reshape(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] > 1e-12:
                w = (w_pair[i, j] + w_pair[j, i]) / 2.0
                if w > 1e-15:
                    grad[i] += w * (pts[i] - pts[j]) / d[i, j]
                    grad[j] += w * (pts[j] - pts[i]) / d[i, j]
    
    return -grad  # We want gradient of the objective (-soft_min)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialize with hex lattice
    pts = _hex_lattice_init(N, r_target=0.053)
    current_score = _compute_score(pts)
    
    best_pts = pts.copy()
    best_score = current_score
    
    # Optimize with L-BFGS-B
    # Use a moderate beta for smoothness
    beta = 30.0
    
    def obj(x):
        return _soft_min_obj(x.reshape(N, 2), beta=beta)
    
    def grad(x):
        return _soft_min_grad(x.reshape(N, 2), beta=beta).flatten()
    
    # Bounds: keep points well inside the unit circle
    bounds = [(-0.95, 0.95)] * (2 * N)
    
    x0 = pts.flatten()
    
    # Try optimization with time check
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B', jac=grad,
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success or result.fun < obj(x0):
            new_pts = result.x.reshape(N, 2)
            new_score = _compute_score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        return best_pts
    
    # Second round with higher beta for sharper min
    beta2 = 80.0
    
    def obj2(x):
        return _soft_min_obj(x.reshape(N, 2), beta=beta2)
    
    def grad2(x):
        return _soft_min_grad(x.reshape(N, 2), beta=beta2).flatten()
    
    try:
        result2 = minimize(
            obj2, best_pts.flatten(), method='L-BFGS-B', jac=grad2,
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result2.success or result2.fun < obj2(best_pts.flatten()):
            new_pts = result2.x.reshape(N, 2)
            new_score = _compute_score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        return best_pts
    
    # Final polish with very high beta
    beta3 = 200.0
    
    def obj3(x):
        return _soft_min_obj(x.reshape(N, 2), beta=beta3)
    
    def grad3(x):
        return _soft_min_grad(x.reshape(N, 2), beta=beta3).flatten()
    
    try:
        result3 = minimize(
            obj3, best_pts.flatten(), method='L-BFGS-B', jac=grad3,
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if result3.success or result3.fun < obj3(best_pts.flatten()):
            new_pts = result3.x.reshape(N, 2)
            new_score = _compute_score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
    except Exception:
        pass
    
    return best_pts
