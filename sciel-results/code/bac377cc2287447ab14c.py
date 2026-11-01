"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_lattice_placement(n, seed=42):
    """Place n points using hexagonal lattice pattern, return array of shape (n, 2)."""
    rng = np.random.RandomState(seed)
    
    # Try different spacings to get at least n points inside unit circle
    best_points = None
    for spacing in np.linspace(0.03, 0.08, 50):
        # Generate hex lattice
        sqrt3 = np.sqrt(3.0)
        dx = spacing
        dy = spacing * sqrt3 / 2.0
        max_coord = 1.2
        xs = np.arange(-max_coord, max_coord + dx, dx)
        ys = np.arange(-max_coord, max_coord + dy, dy)
        
        points = []
        for j, y in enumerate(ys):
            x_offset = (j % 2) * dx / 2.0
            for x in xs:
                px = x + x_offset
                py = y
                if px * px + py * py <= 1.0:
                    points.append([px, py])
        
        points = np.array(points)
        if len(points) >= n:
            # Sort by distance from origin to get compact core
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists, kind='stable')
            best_points = points[idx[:n]]
            break
    
    if best_points is None:
        # Fallback: random placement
        best_points = rng.rand(n, 2) * 2 - 1
        # Keep only inside unit circle
        mask = np.hypot(best_points[:, 0], best_points[:, 1]) <= 1.0
        while mask.sum() < n:
            extra = rng.rand(n, 2) * 2 - 1
            mask2 = np.hypot(extra[:, 0], extra[:, 1]) <= 1.0
            best_points = np.vstack([best_points, extra])
            mask = np.hypot(best_points[:, 0], best_points[:, 1]) <= 1.0
        best_points = best_points[mask][:n]
    
    return best_points


def _compute_r(points):
    """Compute the packing radius r for given points."""
    n = len(points)
    # Boundary constraint
    boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(boundary)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)


def _soft_min_neg(points, beta):
    """Negative soft-min objective: -min(distances, boundary) approximated via log-sum-exp.
    We want to maximize r = min(min_pair_dist/2, min_boundary).
    So we minimize -r.
    
    Using soft-min: -softmin(x) = -(-1/beta * log(sum(exp(-beta*x)))) = (1/beta)*log(sum(exp(-beta*x)))
    """
    n = len(points)
    boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Collect all constraints: boundary distances and half-pair-distances
    # Use soft-min over all of them
    all_vals = np.concatenate([boundary, pair_dists[np.triu_indices(n, k=1)]])
    
    # Soft min: -1/beta * log(sum(exp(-beta * x)))
    # For numerical stability, subtract max
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    soft_min_val = m - np.log(np.sum(exp_vals)) / beta
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val


def _soft_min_neg_grad(points, beta):
    """Gradient of the negative soft-min objective."""
    n = len(points)
    boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    all_vals = np.concatenate([boundary, pair_dists[np.triu_indices(n, k=1)]])
    
    m = np.min(all_vals)
    shifted = all_vals - m
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    # d/dx_i of -soft_min = d/dx_i of (1/beta)*log(sum(exp(-beta*x)))
    # = (1/beta) * sum_j (-beta * exp(-beta*x_j)) / sum(exp(-beta*x_j)) * dx_j/dx_i
    # = -sum_j exp(-beta*x_j)/sum(exp(-beta*x_j)) * dx_j/dx_i
    
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(points)
    
    # Boundary part: x_j = 1 - |p_j|, dx_j/dp_j = -p_j / |p_j|
    for j in range(n):
        r = np.hypot(points[j, 0], points[j, 1])
        if r > 1e-12:
            grad[j] += weights[j] * (-1.0 / r) * points[j]
        # Note: d(-soft_min)/dp_j = -w_j * d(boundary_j)/dp_j = -w_j * (-p_j/r) = w_j * p_j/r
        # Wait, let me redo:
        # objective = -soft_min = (1/beta)*log(sum(exp(-beta*x)))
        # d(obj)/dp_j = (1/beta) * sum_k exp(-beta*x_k)/sum(exp(-beta*x_k)) * (-beta) * dx_k/dp_j
        # = -sum_k w_k * dx_k/dp_j
        # For boundary: x_k = 1 - |p_k|, dx_k/dp_k = -p_k/|p_k|
        # So d(obj)/dp_j from boundary = -w_j * (-p_j/|p_j|) = w_j * p_j/|p_j|
        # But that's for k=j only. For k != j, dx_k/dp_j = 0.
    
    # Let me recompute properly
    grad = np.zeros_like(points)
    
    # Boundary contribution: for each point j, x_j = 1 - |p_j|
    # d(obj)/dp_j includes -w_j * d(x_j)/dp_j = -w_j * (-p_j/|p_j|) = w_j * p_j/|p_j|
    for j in range(n):
        r = np.hypot(points[j, 0], points[j, 1])
        if r > 1e-12:
            grad[j] += weights[j] * points[j] / r
    
    # Pairwise contribution: for pair (i,j) with i<j, x = |p_i - p_j|/2
    # d(obj)/dp_i includes -w * d(x)/dp_i = -w * (p_i - p_j)/|p_i - p_j|
    # d(obj)/dp_j includes -w * d(x)/dp_j = -w * (p_j - p_i)/|p_j - p_i| = -w * (-(p_i-p_j))/|p_i-p_j| = w*(p_i-p_j)/|p_i-p_j|
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    for k in range(len(idx_i)):
        i = idx_i[k]
        j = idx_j[k]
        w = weights[n + k]  # offset by n for pairwise
        dvec = points[i] - points[j]
        dist = np.linalg.norm(dvec)
        if dist > 1e-12:
            unit = dvec / dist
            # x = dist/2, dx/dp_i = unit, dx/dp_j = -unit
            grad[i] += -w * unit
            grad[j] += -w * (-unit)  # = w * unit
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe before 150s kill
    
    # Initial placement
    points = _hex_lattice_placement(N)
    best_points = points.copy()
    best_r = _compute_r(best_points)
    
    # Progressive beta optimization
    betas = [5.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
        
        def objective(x):
            pts = x.reshape(-1, 2)
            return _soft_min_neg(pts, beta)
        
        def grad(x):
            pts = x.reshape(-1, 2)
            g = _soft_min_neg_grad(pts, beta)
            return g.flatten()
        
        x0 = best_points.flatten()
        
        try:
            res = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        except Exception:
            continue
        
        new_points = res.x.reshape(-1, 2)
        
        # Validate
        if not np.all(np.isfinite(new_points)):
            continue
        if np.any(np.hypot(new_points[:, 0], new_points[:, 1]) > 1.0):
            continue
        
        new_r = _compute_r(new_points)
        if new_r > best_r:
            best_r = new_r
            best_points = new_points.copy()
    
    # Final polish with very high beta
    if time.time() < deadline - 5.0:
        beta = 2000.0
        
        def objective(x):
            pts = x.reshape(-1, 2)
            return _soft_min_neg(pts, beta)
        
        def grad(x):
            pts = x.reshape(-1, 2)
            g = _soft_min_neg_grad(pts, beta)
            return g.flatten()
        
        x0 = best_points.flatten()
        
        try:
            res = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 300, 'maxfun': 800, 'ftol': 1e-12, 'gtol': 1e-10})
            new_points = res.x.reshape(-1, 2)
            if np.all(np.isfinite(new_points)) and np.all(np.hypot(new_points[:, 0], new_points[:, 1]) <= 1.0):
                new_r = _compute_r(new_points)
                if new_r > best_r:
                    best_r = new_r
                    best_points = new_points.copy()
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
