"""Rewritten program: force-directed relaxation with soft-min objective for circle packing."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 115.0  # safety margin before 120s limit


def _soft_min_dist(points, beta):
    """Compute soft-min of pairwise distances and boundary distances.
    
    Returns (value, gradient) where value approximates min distance.
    """
    n = len(points)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-30)
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances: 1 - |c_i|
    r = np.sqrt(np.sum(points ** 2, axis=1))
    boundary = 1.0 - r
    
    # Combine all constraints
    # We want min of: dist[i,j] for i<j, and boundary[i]
    # For soft-min, we use log-sum-exp trick
    
    # Gather all distances into a flat array
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]
    all_dists = np.concatenate([pair_dists, boundary])
    
    # Soft-min: -log(sum(exp(-beta * x))) / beta
    # For numerical stability, subtract max
    max_val = np.max(all_dists)
    exp_terms = np.exp(-beta * (all_dists - max_val))
    soft_min_val = max_val + np.log(np.sum(exp_terms)) / beta
    
    # Gradient
    # d(soft_min)/d(x_k) = -exp(-beta * x_k) / sum(exp(-beta * x_j)) * beta... 
    # Actually: soft_min = max_val + log(sum(exp(-beta*(x_j - max_val))))/beta
    # d(soft_min)/d(x_k) = -exp(-beta*(x_k - max_val)) / (beta * sum(exp(-beta*(x_j - max_val))))
    
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros_like(points)
    # Pairwise gradients
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = weights[idx]
        # d(dist_ij)/d(points_i) = (points_i - points_j) / dist_ij
        d_ij = diff[i, j] / dist[i, j]
        grad[i] += -w * d_ij
        grad[j] += w * d_ij
    
    # Boundary gradients
    for i in range(n):
        w = weights[n * (n - 1) // 2 + i]
        # d(boundary_i)/d(points_i) = -points_i / |points_i|
        if r[i] > 1e-12:
            grad[i] += w * (-points[i] / r[i])
        else:
            grad[i] += w * np.array([1.0, 0.0])  # arbitrary direction
    
    return soft_min_val, grad


def _objective_and_grad(x, beta):
    """Objective to maximize: soft-min of all distances. We minimize negative."""
    points = x.reshape(-1, 2)
    val, grad = _soft_min_dist(points, beta)
    return -val, -grad.reshape(-1)


def _construct_hex_init():
    """Generate hexagonal lattice initialization."""
    spacing = 0.22
    rows = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                rows.append([x, y])
    points = np.array(rows)
    # Select N closest to center
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r)[:N]
    return points[idx]


def _construct_ring_init():
    """Generate ring-based initialization."""
    points = [[0.0, 0.0]]
    # First ring: 6 circles
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        points.append([0.22 * np.cos(angle), 0.22 * np.sin(angle)])
    # Second ring: 12 circles
    for angle in np.linspace(0.26, 2 * np.pi + 0.26, 12, endpoint=False):
        points.append([0.44 * np.cos(angle), 0.44 * np.sin(angle)])
    # Third ring: 18 circles
    for angle in np.linspace(0.17, 2 * np.pi + 0.17, 18, endpoint=False):
        points.append([0.66 * np.cos(angle), 0.66 * np.sin(angle)])
    
    points = np.array(points[:N])
    return points


def _construct_random_init(rng):
    """Generate random initialization inside unit disk."""
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = rng.uniform(0, 0.9, N)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def _score(points):
    """Compute the actual score: min of all pairwise distances/2 and boundary distances."""
    n = len(points)
    if n != N:
        return 0.0
    
    # Check finite
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check inside container
    r = np.sqrt(np.sum(points ** 2, axis=1))
    if np.any(r >= 1.0):
        return 0.0
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    min_pair = np.min(dist) / 2.0
    min_boundary = np.min(1.0 - r)
    
    return min(min_pair, min_boundary)


def construct():
    start_time = time.time()
    
    # Try multiple initializations and keep the best result
    best_points = None
    best_score = 0.0
    
    inits = [
        _construct_hex_init,
        _construct_ring_init,
        lambda: _construct_random_init(np.random.default_rng(42)),
        lambda: _construct_random_init(np.random.default_rng(123)),
        lambda: _construct_random_init(np.random.default_rng(999)),
    ]
    
    # Beta schedule: start with lower beta for smooth optimization, increase for precision
    beta_schedule = [10.0, 50.0, 200.0, 1000.0, 5000.0]
    
    for init_func in inits:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        try:
            points = init_func()
        except Exception:
            continue
        
        # Optimize with increasing beta
        current_points = points.copy()
        
        for beta in beta_schedule:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            try:
                result = minimize(
                    _objective_and_grad,
                    current_points.flatten(),
                    args=(beta,),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                current_points = result.x.reshape(-1, 2)
            except Exception:
                break
        
        # Final high-precision polish
        if time.time() - start_time < DEADLINE - 3:
            try:
                result = minimize(
                    _objective_and_grad,
                    current_points.flatten(),
                    args=(100000.0,),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                current_points = result.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Evaluate actual score
        score = _score(current_points)
        if score > best_score:
            best_score = score
            best_points = current_points.copy()
    
    # If we didn't get a valid result, fall back to hex lattice
    if best_points is None or best_score <= 0:
        best_points = _construct_hex_init()
    
    # Final validation
    if not np.all(np.isfinite(best_points)) or len(best_points) != N:
        best_points = _construct_hex_init()
    
    return best_points
