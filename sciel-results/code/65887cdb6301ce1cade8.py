"""Rewritten program for packing 300 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 300
DEADLINE = 110.0  # seconds, safety margin before 150s limit


def _hex_lattice_init(n):
    """Generate a hexagonal lattice initialization for n points in unit disk."""
    spacing = np.sqrt(np.pi / n) * 0.95
    
    rows = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    radius = radius[mask]
    
    if len(points) < n:
        theta = np.linspace(0, 2 * np.pi * np.sqrt(n), n)
        r = np.sqrt(np.linspace(0, 1, n))
        points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    else:
        idx = np.argsort(radius, kind='stable')[:n]
        points = points[idx]
    
    return points


def _radial_ring_init(n):
    """Generate a radial ring initialization: points placed on concentric rings."""
    points = []
    # Start with center
    points.append([0.0, 0.0])
    remaining = n - 1
    ring = 1
    while remaining > 0:
        # Number of points on this ring
        # For a hexagonal-like packing, ring k has 6*k points
        ring_count = min(6 * ring, remaining)
        r = ring * np.sqrt(np.pi / n) * 0.95
        # Adjust radius to fit within unit circle
        r = min(r, 1.0 - 0.05)
        angles = np.linspace(0, 2 * np.pi, ring_count, endpoint=False) + np.pi / ring_count
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
        remaining -= ring_count
        ring += 1
    
    return np.array(points[:n])


def _compute_score(points):
    """Compute the packing score: min of half-min-distance and min-margin-to-boundary."""
    n = len(points)
    if n < 2:
        return 0.0
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    r = np.sqrt(np.sum(points ** 2, axis=1))
    min_margin = np.min(1.0 - r)
    
    return min(min_dist / 2.0, min_margin)


def _soft_min_obj_and_grad(points, beta=100.0):
    """
    Compute soft-min objective and gradient simultaneously.
    Returns (obj_value, grad_array) where grad is the gradient of the objective
    (i.e., gradient of -soft_min, so we minimize this).
    """
    n = len(points)
    if n < 2:
        return 0.0, np.zeros_like(points)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(dist_sq, np.inf)
    
    # All values for soft-min
    # We use only upper triangle to avoid double counting
    # pair_vals[i,j] = dist[i,j] / 2 for i != j
    # margin_vals[i] = 1 - |points[i]|
    
    r = np.sqrt(np.sum(points ** 2, axis=1))
    margin_vals = 1.0 - r
    
    # Build the full set of values
    # Use upper triangle for pairs to avoid redundancy
    iu = np.triu_indices(n, k=1)
    pair_vals = dist[iu] / 2.0
    
    all_vals = np.concatenate([pair_vals, margin_vals])
    
    # Soft-min computation
    max_val = np.max(all_vals)
    shifted = all_vals - max_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min = max_val - log_sum / beta
    obj_val = -soft_min
    
    # Weights
    weights = exp_terms / sum_exp
    
    # Gradient computation
    grad = np.zeros_like(points)
    
    # Pairwise gradient: for pair (i,j), weight w, gradient contribution to point i is
    # w * (points[i] - points[j]) / dist[i,j]
    # And to point j is w * (points[j] - points[i]) / dist[i,j] = -w * (points[i] - points[j]) / dist[i,j]
    # Since we're using upper triangle, for each (i,j) with i<j:
    # grad[i] += w * (points[i] - points[j]) / dist[i,j]
    # grad[j] -= w * (points[i] - points[j]) / dist[i,j]
    
    pair_weights = weights[:len(pair_vals)]
    ii, jj = iu
    d = dist[ii, jj]
    safe_d = np.where(d > 1e-12, d, 1.0)
    unit_vec = diff[ii, jj] / safe_d[:, np.newaxis]
    
    # Accumulate gradients
    np.add.at(grad, ii, pair_weights[:, np.newaxis] * unit_vec)
    np.add.at(grad, jj, -pair_weights[:, np.newaxis] * unit_vec)
    
    # Boundary gradient: for point i, margin = 1 - r_i
    # d(margin)/d(points[i]) = -points[i] / r_i
    # So gradient of -soft_min w.r.t. points[i] from boundary term is:
    # w_i * (-1) * d(margin)/d(points[i]) = w_i * points[i] / r_i
    # Wait, let me think again.
    # soft_min = min of all vals. We minimize -soft_min.
    # d(-soft_min)/d(points[i]) = -d(soft_min)/d(points[i])
    # d(soft_min)/d(v_k) where v_k is the k-th value = -w_k (from log-sum-exp derivative)
    # Actually: soft_min = max_val - log(sum(exp(-beta*(v - max_val))))/beta
    # d(soft_min)/d(v_k) = -(-beta * exp(-beta*(v_k-max_val))) / (beta * sum(exp(...))) = w_k
    # Wait no. Let me redo.
    # soft_min = max_val - (1/beta) * log(sum_k exp(-beta*(v_k - max_val)))
    # d(soft_min)/d(v_k) = -(1/beta) * (-beta * exp(-beta*(v_k-max_val))) / sum_j exp(-beta*(v_j-max_val))
    # = exp(-beta*(v_k-max_val)) / sum_j exp(...) = w_k
    # So d(soft_min)/d(v_k) = w_k
    # And v_k depends on points.
    # For pair (i,j): v = dist(i,j)/2, d(v)/d(points[i]) = (points[i]-points[j])/dist(i,j)
    # For boundary i: v = 1-r_i, d(v)/d(points[i]) = -points[i]/r_i
    # d(-soft_min)/d(points[i]) = -sum_k w_k * d(v_k)/d(points[i])
    
    # So:
    # For pairs: grad[i] += -w_ij * (points[i]-points[j])/dist_ij  (for each j)
    # For boundary: grad[i] += -w_i * (-points[i]/r_i) = w_i * points[i]/r_i
    
    # But wait, I already computed the pairwise part as:
    # grad[i] += w * (points[i]-points[j])/dist  and grad[j] -= w*(points[i]-points[j])/dist
    # That's d(soft_min)/d(points), not d(-soft_min)/d(points)
    # So I need to negate the pairwise part and add the boundary part with correct sign.
    
    # Let me restart the gradient:
    grad = np.zeros_like(points)
    
    # Pairwise: d(-soft_min)/d(points[i]) = -sum_j w_ij * (points[i]-points[j])/dist_ij
    # Using upper triangle: for (i,j) i<j:
    #   contribution to grad[i]: -w_ij * (points[i]-points[j])/dist_ij
    #   contribution to grad[j]: -w_ij * (points[j]-points[i])/dist_ij = w_ij * (points[i]-points[j])/dist_ij
    
    np.add.at(grad, ii, -pair_weights[:, np.newaxis] * unit_vec)
    np.add.at(grad, jj, pair_weights[:, np.newaxis] * unit_vec)
    
    # Boundary: d(-soft_min)/d(points[i]) = -w_i * d(1-r_i)/d(points[i]) = -w_i * (-points[i]/r_i) = w_i * points[i]/r_i
    margin_weights = weights[len(pair_vals):]
    safe_r = np.where(r > 1e-12, r, 1.0)
    grad += margin_weights[:, np.newaxis] * points / safe_r[:, np.newaxis]
    
    return obj_val, grad


def _gd_polish(points, beta=200.0, max_iter=200, time_budget=10.0):
    """Fast gradient descent polish with adaptive step size."""
    start_time = time.time()
    pts = points.copy()
    lr = 0.01
    best_obj = _soft_min_obj_and_grad(pts, beta=beta)[0]
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        obj, grad = _soft_min_obj_and_grad(pts, beta=beta)
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Adaptive step size
        step = lr / (grad_norm + 1e-10)
        new_pts = pts - step * grad
        
        # Check validity
        r = np.sqrt(np.sum(new_pts ** 2, axis=1))
        if not np.all(r < 1.0) or not np.all(np.isfinite(new_pts)):
            lr *= 0.5
            continue
        
        new_obj = _soft_min_obj_and_grad(new_pts, beta=beta)[0]
        if new_obj < obj:
            pts = new_pts
            lr = min(lr * 1.1, 0.1)
        else:
            lr *= 0.7
    
    return pts if np.all(np.isfinite(pts)) else points


def _optimize_lbfgs(points, beta, max_iter=100, time_budget=15.0):
    """Optimize using L-BFGS-B with given beta."""
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > time_budget:
            raise TimeoutError()
        pts = x.reshape(-1, 2)
        return _soft_min_obj_and_grad(pts, beta=beta)[0]
    
    def gradient(x):
        if time.time() - start_time > time_budget:
            raise TimeoutError()
        pts = x.reshape(-1, 2)
        return _soft_min_obj_and_grad(pts, beta=beta)[1].ravel()
    
    x0 = points.ravel()
    
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B', jac=gradient,
            options={'maxiter': max_iter, 'maxfun': max_iter * 5, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        new_points = result.x.reshape(-1, 2)
        if np.all(np.isfinite(new_points)):
            r = np.sqrt(np.sum(new_points ** 2, axis=1))
            if np.all(r < 1.0):
                return new_points
    except Exception:
        pass
    
    return points


def construct():
    start_time = time.time()
    
    # Try multiple initializations
    inits = [_hex_lattice_init(N), _radial_ring_init(N)]
    
    best_score = -1.0
    best_points = None
    
    for init_pts in inits:
        if time.time() - start_time > DEADLINE - 30:
            break
        score = _compute_score(init_pts)
        if score > best_score:
            best_score = score
            best_points = init_pts.copy()
    
    # Beta annealing schedule
    betas = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 2000.0, 5000.0, 20000.0]
    
    for beta in betas:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # L-BFGS optimization
        new_points = _optimize_lbfgs(best_points, beta=beta, max_iter=80, time_budget=12.0)
        score = _compute_score(new_points)
        if score > best_score:
            best_score = score
            best_points = new_points.copy()
        
        # GD polish
        if time.time() - start_time > DEADLINE - 10:
            break
        polished = _gd_polish(best_points, beta=beta, max_iter=100, time_budget=8.0)
        score = _compute_score(polished)
        if score > best_score:
            best_score = score
            best_points = polished.copy()
    
    # Final high-precision polish with very high beta
    if time.time() - start_time < DEADLINE - 5:
        final = _gd_polish(best_points, beta=100000.0, max_iter=200, time_budget=8.0)
        score = _compute_score(final)
        if score > best_score:
            best_score = score
            best_points = final.copy()
    
    # Ensure validity
    r = np.sqrt(np.sum(best_points ** 2, axis=1))
    if not np.all(r < 1.0) or not np.all(np.isfinite(best_points)):
        best_points = _hex_lattice_init(N)
    
    return best_points


if __name__ == '__main__':
    result = construct()
    print(f"Score: {_compute_score(result)}")
    print(f"Shape: {result.shape}")
