"""Rewritten packing solver: hexagonal init + L-BFGS-B smooth max-min objective."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200
DEADLINE = 120.0  # wall-clock limit for construct()


def _score(pts):
    """Compute the common radius for a set of centres in [-0.5, 0.5]^2."""
    # Boundary distance
    d = np.array([
        0.5 - np.abs(pts[:, 0]),
        0.5 - np.abs(pts[:, 1])
    ])
    min_boundary = np.min(d, axis=1)

    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist, axis=1)

    return np.min(np.minimum(min_boundary, min_pair / 2.0))


def _objective_and_grad(pts_flat):
    """
    Smooth surrogate for max-min radius using log-sum-exp.
    We maximize r = min(min_pair/2, min_boundary).
    Equivalently minimize -r.
    
    Using a soft-min: r_soft = -logsumexp(-d_i / tau) * tau
    where d_i are all the candidate distances (pairwise/2 and boundary).
    
    For gradient: d(r_soft)/d(pts) = sum(d_i/tau * exp(-d_i/tau) / sum(exp(-d_i/tau))) * ...
    Let's use a simpler approach: minimize the sum of squared violations + maximize radius.
    
    Actually, let's use the standard approach: minimize -min(distances) with a smooth approximation.
    f(x) = -soft_min(d) where soft_min(d) = -tau * log(sum(exp(-d_i/tau)))
    so f(x) = tau * log(sum(exp(-d_i/tau)))
    
    grad f = tau * [sum(exp(-d_i/tau) * (-1/tau) * grad d_i) / sum(exp(-d_i/tau))]
           = -sum(exp(-d_i/tau) * grad d_i) / sum(exp(-d_i/tau))
           = -sum(w_i * grad d_i) where w_i = exp(-d_i/tau) / sum(exp(-d_j/tau))
    """
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    
    # Compute all distances
    # Pairwise: for i < j, d_ij = |c_i - c_j| / 2
    # Boundary: for each i, d_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    
    # To keep memory manageable, compute in chunks or use broadcasting carefully
    # For N=200, pairwise matrix is 200x200 = 40000 entries, fine.
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_pair = np.sqrt(np.sum(diff * diff, axis=-1)) / 2.0
    np.fill_diagonal(dist_pair, np.inf)
    
    # Boundary distances
    bx = 0.5 - np.abs(pts[:, 0])
    by = 0.5 - np.abs(pts[:, 1])
    dist_boundary = np.minimum(bx, by)
    
    # Combine all distances into a flat array for soft-min
    # We'll use a moderate tau
    tau = 0.002
    
    # For numerical stability, subtract max
    all_dists = np.concatenate([dist_pair.ravel(), dist_boundary.ravel()])
    all_dists = all_dists[np.isfinite(all_dists)]
    
    # soft_min = -tau * log(sum(exp(-d/tau)))
    # f = -soft_min = tau * log(sum(exp(-d/tau)))
    
    max_d = np.max(all_dists)
    exp_vals = np.exp(-(all_dists - max_d) / tau)
    sum_exp = np.sum(exp_vals)
    
    f = tau * (np.log(sum_exp) - max_d / tau)
    
    # Gradient computation
    # grad f = -sum(w_i * grad d_i) where w_i = exp(-(d_i - max_d)/tau) / sum_exp
    # But we need grad of each distance w.r.t. each point.
    
    # This is complex. Let's use a simpler gradient-free approach or finite differences?
    # No, L-BFGS-B needs gradients. Let's compute them properly.
    
    # Actually, let's use a different approach: use the fact that for large enough separation,
    # the min is achieved by a few pairs. We can use a "k-min" approximation.
    
    # Alternative: use scipy's minimize with method='L-BFGS-B' and provide analytic gradient.
    # For now, let's use a simpler objective: minimize sum of (r - d_i)^2 for d_i < r,
    # but that requires knowing r.
    
    # Let's go back to the log-sum-exp approach and compute gradients carefully.
    
    # The gradient of f = tau * log(sum(exp(-(d_i - max_d)/tau))) w.r.t. x
    # = tau * [1/sum(exp(...)) * sum(exp(-(d_i - max_d)/tau) * (-1/tau) * grad d_i)]
    # = -sum(exp(-(d_i - max_d)/tau) * grad d_i) / sum(exp(-(d_j - max_d)/tau))
    # = -sum(w_i * grad d_i)
    
    # We need grad d_i for each distance.
    # For pairwise distance d_ij = |c_i - c_j|/2:
    #   grad_{c_i} d_ij = (c_i - c_j) / (2 * |c_i - c_j|)
    #   grad_{c_j} d_ij = (c_j - c_i) / (2 * |c_i - c_j|)
    # For boundary distance d_i = min(bx_i, by_i):
    #   if bx_i < by_i: grad = (-sign(x_i), 0)
    #   if by_i < bx_i: grad = (0, -sign(y_i))
    #   if equal: average
    
    # This is O(N^2) per gradient evaluation, which is fine for N=200.
    
    w = exp_vals / sum_exp
    
    grad_flat = np.zeros_like(pts_flat)
    
    # Pairwise contributions
    # w_pair corresponds to the first n*(n-1)/2 entries of all_dists
    # But we flattened dist_pair which is n x n, so the order is row-major.
    # dist_pair[i,j] for i != j
    # Let's reconstruct which weights correspond to which pairs.
    
    # Actually, let's just compute the gradient by accumulating.
    w_pair = w[:n*n]  # This includes diagonal entries which are inf, so exp is 0
    # But we set diagonal to inf, so exp(-(inf-max_d)/tau) = exp(-inf) = 0
    w_pair = w_pair.reshape(n, n)
    
    # For each pair (i, j) with i < j, the distance appears twice in the matrix: (i,j) and (j,i)
    # Both have the same weight.
    for i in range(n):
        for j in range(i+1, n):
            w_ij = w_pair[i, j] + w_pair[j, i]
            if w_ij < 1e-15:
                continue
            diff_ij = pts[i] - pts[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij < 1e-15:
                continue
            grad_dir = diff_ij / dist_ij  # grad of |c_i - c_j| w.r.t. c_i
            # d_ij = |c_i - c_j| / 2, so grad d_ij = grad_dir / 2
            grad_flat[2*i] -= w_ij * grad_dir[0] / 2.0
            grad_flat[2*i+1] -= w_ij * grad_dir[1] / 2.0
            grad_flat[2*j] += w_ij * grad_dir[0] / 2.0
            grad_flat[2*j+1] += w_ij * grad_dir[1] / 2.0
    
    # Boundary contributions
    w_bound = w[n*n:]
    for i in range(n):
        w_i = w_bound[i]
        if w_i < 1e-15:
            continue
        x_i = pts[i, 0]
        y_i = pts[i, 1]
        bx_i = 0.5 - np.abs(x_i)
        by_i = 0.5 - np.abs(y_i)
        if bx_i < by_i:
            grad_flat[2*i] += w_i * (-np.sign(x_i))
            grad_flat[2*i+1] += 0.0
        elif by_i < bx_i:
            grad_flat[2*i] += 0.0
            grad_flat[2*i+1] += w_i * (-np.sign(y_i))
        else:
            grad_flat[2*i] += w_i * (-0.5 * np.sign(x_i))
            grad_flat[2*i+1] += w_i * (-0.5 * np.sign(y_i))
    
    return f, grad_flat


def _hex_init(n):
    """Generate a hexagonal lattice initialization for n points in [-0.5, 0.5]^2."""
    # Hexagonal packing: spacing s, rows offset
    # Area per point ~ s^2 * sqrt(3)/2
    # Total area = 1, so s ~ sqrt(2 / (n * sqrt(3)))
    s = np.sqrt(2.0 / (n * np.sqrt(3.0)))
    
    # Generate hex grid points
    points = []
    # Estimate number of rows
    n_rows = int(np.ceil(np.sqrt(n * 2.0 / np.sqrt(3.0))))
    
    for row in range(n_rows * 2):
        y = (row - n_rows) * s * np.sqrt(3.0) / 2.0
        offset = (row % 2) * s / 2.0
        for col in range(int(np.ceil(1.0 / s)) + 2):
            x = (col - int(np.ceil(1.0 / s)) / 2.0) * s + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n:
        # Fill remaining with random points
        while len(points) < n:
            points = np.vstack([points, np.random.uniform(-0.45, 0.45, size=(1, 2))])
    else:
        # Randomly select n points
        idx = np.random.choice(len(points), n, replace=False)
        points = points[idx]
    
    return points


def construct():
    start_time = time.time()
    
    # Hexagonal initialization
    pts = _hex_init(N)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # L-BFGS-B optimization
    x0 = pts.ravel()
    
    def obj(x):
        return _objective_and_grad(x)[0]
    
    def obj_and_grad(x):
        return _objective_and_grad(x)
    
    # Run optimization with time check
    try:
        result = minimize(
            obj_and_grad,
            x0,
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-10,
                'gtol': 1e-8,
                'maxls': 20,
            }
        )
        
        if result.success or result.fun < obj(x0):
            new_pts = result.x.reshape(-1, 2)
            # Clip to valid range
            new_pts = np.clip(new_pts, -0.5, 0.5)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return best_pts
    
    # Second pass with smaller tau for sharper min
    try:
        global _tau
        _tau = 0.0005  # Sharper
        
        # Redefine with smaller tau
        def obj2(x):
            return _objective_and_grad_tau(x, 0.0005)[0]
        
        def obj2_grad(x):
            return _objective_and_grad_tau(x, 0.0005)
        
        x0_2 = best_pts.ravel()
        result2 = minimize(
            obj2_grad,
            x0_2,
            method='L-BFGS-B',
            options={
                'maxiter': 300,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'maxls': 20,
            }
        )
        
        if result2.success or result2.fun < obj2(x0_2):
            new_pts2 = result2.x.reshape(-1, 2)
            new_pts2 = np.clip(new_pts2, -0.5, 0.5)
            new_score2 = _score(new_pts2)
            if new_score2 > best_score:
                best_score = new_score2
                best_pts = new_pts2
    except Exception:
        pass
    
    return best_pts


def _objective_and_grad_tau(pts_flat, tau):
    """Smooth max-min objective with specified tau."""
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_pair = np.sqrt(np.sum(diff * diff, axis=-1)) / 2.0
    np.fill_diagonal(dist_pair, np.inf)
    
    bx = 0.5 - np.abs(pts[:, 0])
    by = 0.5 - np.abs(pts[:, 1])
    dist_boundary = np.minimum(bx, by)
    
    all_dists = np.concatenate([dist_pair.ravel(), dist_boundary.ravel()])
    all_dists = all_dists[np.isfinite(all_dists)]
    
    max_d = np.max(all_dists)
    exp_vals = np.exp(-(all_dists - max_d) / tau)
    sum_exp = np.sum(exp_vals)
    
    f = tau * (np.log(sum_exp) - max_d / tau)
    
    w = exp_vals / sum_exp
    
    grad_flat = np.zeros_like(pts_flat)
    
    w_pair = w[:n*n].reshape(n, n)
    
    for i in range(n):
        for j in range(i+1, n):
            w_ij = w_pair[i, j] + w_pair[j, i]
            if w_ij < 1e-15:
                continue
            diff_ij = pts[i] - pts[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij < 1e-15:
                continue
            grad_dir = diff_ij / dist_ij
            grad_flat[2*i] -= w_ij * grad_dir[0] / 2.0
            grad_flat[2*i+1] -= w_ij * grad_dir[1] / 2.0
            grad_flat[2*j] += w_ij * grad_dir[0] / 2.0
            grad_flat[2*j+1] += w_ij * grad_dir[1] / 2.0
    
    w_bound = w[n*n:]
    for i in range(n):
        w_i = w_bound[i]
        if w_i < 1e-15:
            continue
        x_i = pts[i, 0]
        y_i = pts[i, 1]
        bx_i = 0.5 - np.abs(x_i)
        by_i = 0.5 - np.abs(y_i)
        if bx_i < by_i:
            grad_flat[2*i] += w_i * (-np.sign(x_i))
        elif by_i < bx_i:
            grad_flat[2*i+1] += w_i * (-np.sign(y_i))
        else:
            grad_flat[2*i] += w_i * (-0.5 * np.sign(x_i))
            grad_flat[2*i+1] += w_i * (-0.5 * np.sign(y_i))
    
    return f, grad_flat
