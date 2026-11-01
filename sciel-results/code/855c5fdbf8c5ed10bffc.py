"""Rewritten program for packing 300 equal circles in a unit circle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300
_DEADLINE = 148.0  # seconds (2s safety margin before 150s kill)
_START_TIME = time.time()


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _compute_score(points):
    """Compute packing score: min(half_min_pairwise_dist, min_distance_to_boundary)."""
    n = len(points)
    if n < 2:
        return 0.0
    
    # All pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pairwise = np.sqrt(np.min(dists_sq))
    
    # Distance to boundary
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_pairwise / 2.0, min_boundary)


def _soft_min_score(points, beta):
    """Smooth soft-min surrogate for the packing score.
    
    The score is min(d_ij/2 for all pairs, 1-|c_i| for all i).
    We use a log-sum-exp based soft minimum to make it differentiable.
    """
    n = len(points)
    
    # All pairwise half-distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    half_dists = 0.5 * np.sqrt(np.maximum(dists_sq, 0))
    
    # Boundary distances
    radii = np.hypot(points[:, 0], points[:, 1])
    boundary_dists = 1.0 - radii
    
    # Combine all constraints
    all_dists = np.concatenate([half_dists.ravel(), boundary_dists.ravel()])
    all_dists = np.maximum(all_dists, 1e-15)  # avoid log(0)
    
    # Soft minimum using log-sum-exp trick
    # soft_min(x) = -logsumexp(-beta*x)/beta
    # = -1/beta * log(sum(exp(-beta*x)))
    # For numerical stability:
    m = np.min(all_dists)
    shifted = -beta * (all_dists - m)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = m - log_sum / beta
    
    return soft_min


def _soft_min_grad(points, beta):
    """Compute gradient of the soft-min score with respect to points."""
    n = len(points)
    dim = 2
    
    # All pairwise half-distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    half_dists = 0.5 * np.sqrt(np.maximum(dists_sq, 0))
    half_dists_flat = half_dists.ravel().copy()
    np.fill_diagonal(half_dists, np.inf)
    
    # Boundary distances
    radii = np.hypot(points[:, 0], points[:, 1])
    boundary_dists = 1.0 - radii
    
    # Combine all constraints
    all_dists = np.concatenate([half_dists_flat, boundary_dists.ravel()])
    all_dists = np.maximum(all_dists, 1e-15)
    
    # Compute weights for soft-min
    shifted = -beta * (all_dists - np.min(all_dists))
    shifted = np.clip(shifted, -500, 500)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(points)
    
    # Number of pairwise constraints
    n_pairs = n * (n - 1) // 2
    
    # Gradient for pairwise constraints
    for idx in range(n_pairs):
        i = idx // (n - 1)
        j = idx % (n - 1)
        if j >= i:
            j += 1
        
        w = weights[idx]
        d = half_dists[i, j]
        if d > 1e-10:
            diff = points[i] - points[j]
            unit = diff / (2.0 * d)  # d/2, so grad of d/2 is diff/(2d)
            # Actually: half_dist = 0.5 * |p_i - p_j|
            # d(half_dist)/dp_i = 0.5 * (p_i - p_j) / |p_i - p_j|
            # d(half_dist)/dp_j = -0.5 * (p_i - p_j) / |p_i - p_j|
            unit = diff / (2.0 * 2.0 * d)  # 0.5 * diff / (2*d) = diff/(4d)? No...
            # half_dist = 0.5 * dist
            # grad of half_dist wrt p_i = 0.5 * (p_i-p_j)/dist = 0.5 * diff/d
            unit = 0.5 * diff / d
            grad[i] += w * unit
            grad[j] -= w * unit
    
    # Gradient for boundary constraints
    for i in range(n):
        w = weights[n_pairs + i]
        r = radii[i]
        if r > 1e-10:
            # boundary_dist = 1 - |p_i|
            # grad wrt p_i = -p_i/|p_i|
            grad[i] -= w * points[i] / r
    
    return grad


def _hex_lattice_init(n):
    """Generate initial hexagonal lattice points inside unit disk."""
    # Estimate spacing
    spacing = 1.8 / (np.sqrt(n) + 0.5)
    
    points = []
    for k in range(-25, 26):
        for j in range(-25, 26):
            x = (k + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r < 0.92:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) >= n:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists, kind='stable')[:n]
        points = points[idx]
    else:
        while len(points) < n:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 0.9)
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
    
    return points


def _concentric_rings_init(n):
    """Alternative initialization with concentric rings."""
    points = [[0.0, 0.0]]
    remaining = n - 1
    radius = 0.15
    
    while remaining > 0:
        # Estimate how many fit on this ring
        circumference = 2 * np.pi * radius
        count = int(circumference / (2.0 * radius * 0.35))
        count = min(count, remaining)
        if count < 1:
            count = 1
        
        for k in range(count):
            angle = 2 * np.pi * k / count
            points.append([radius * np.cos(angle), radius * np.sin(angle)])
        
        remaining -= count
        radius += 0.08
    
    points = np.array(points[:n])
    return points


def _lbfgs_optimize(points, beta, max_iter=200):
    """Run L-BFGS-B optimization on soft-min score."""
    
    def objective(x):
        pts = x.reshape(-1, 2)
        return -_soft_min_score(pts, beta)  # Minimize negative score
    
    def gradient(x):
        pts = x.reshape(-1, 2)
        g = _soft_min_grad(pts, beta)
        return -g.ravel()  # Gradient of negative objective
    
    x0 = points.ravel()
    
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': 1e-6, 'ftol': 1e-10}
    )
    
    return result.x.reshape(-1, 2), result.success


def _project_inside_unit_disk(points):
    """Ensure all points are strictly inside the unit circle."""
    radii = np.hypot(points[:, 0], points[:, 1])
    for i in range(len(points)):
        if radii[i] >= 0.999:
            points[i] = points[i] * 0.998 / radii[i]
    return points


def construct():
    # Try multiple initializations and optimizations
    best_points = None
    best_score = -np.inf
    
    inits = [
        _hex_lattice_init(N),
        _concentric_rings_init(N),
    ]
    
    # Add a random initialization
    random_pts = np.random.uniform(-0.9, 0.9, size=(N, 2))
    random_pts = _project_inside_unit_disk(random_pts)
    inits.append(random_pts)
    
    for init_pts in inits:
        if _remaining_time() < 15.0:
            break
        
        current_pts = init_pts.copy()
        current_score = _compute_score(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts.copy()
        
        # Multi-stage beta annealing
        betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
        
        for beta in betas:
            if _remaining_time() < 10.0:
                break
            
            # Run L-BFGS-B
            opt_pts, success = _lbfgs_optimize(current_pts, beta, max_iter=150)
            
            # Project inside disk
            opt_pts = _project_inside_unit_disk(opt_pts)
            
            opt_score = _compute_score(opt_pts)
            
            if opt_score > current_score:
                current_pts = opt_pts
                current_score = opt_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
        
        # Final polish with high beta
        if _remaining_time() > 8.0:
            for beta in [300.0, 500.0]:
                if _remaining_time() < 5.0:
                    break
                opt_pts, success = _lbfgs_optimize(current_pts, beta, max_iter=100)
                opt_pts = _project_inside_unit_disk(opt_pts)
                opt_score = _compute_score(opt_pts)
                if opt_score > current_score:
                    current_pts = opt_pts
                    current_score = opt_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
    
    # Final validation
    best_points = _project_inside_unit_disk(best_points)
    final_score = _compute_score(best_points)
    
    return best_points


# EVOLVE-BLOCK-END
