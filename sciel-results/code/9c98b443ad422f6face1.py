"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _softmin_logsumexp(x, beta):
    """Soft-min of x using log-sum-exp trick. x: 1-D array."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    m = np.min(x)
    # softmin = m - (1/beta) * log(sum(exp(-beta*(x - m))))
    z = -beta * (x - m)
    # clip for numerical stability
    z = np.clip(z, -500, 500)
    s = np.sum(np.exp(z))
    if s <= 0:
        return m
    return m - (1.0 / beta) * np.log(s)


def _score(points):
    """Compute the packing radius for a set of points."""
    n = len(points)
    if n == 0:
        return 0.0
    # pairwise distances
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # boundary
    radii = np.sqrt(np.sum(points * points, axis=1))
    min_bound = 1.0 - np.min(radii)
    return min(min_pair, min_bound)


def _grad_softmin(points, beta):
    """
    Compute gradient of soft-min objective w.r.t. points.
    Objective: softmin of all pairwise distances/2 and (1 - |c_i|).
    We maximize this, so we want gradient of the soft-min.
    """
    n = len(points)
    # Build list of all constraint values
    # Pairwise: for i<j, d_ij/2
    # Boundary: 1 - |c_i|
    
    # Compute all pairwise distances
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # Soft-min over pairwise distances/2 and boundary terms
    # Let's build a vector of all constraints
    # Number of pairwise: n*(n-1)/2
    # Number of boundary: n
    n_pairs = n * (n - 1) // 2
    total = n_pairs + n
    
    # Values
    vals = np.empty(total)
    vals[:n_pairs] = dist[np.triu_indices(n, k=1)] / 2.0
    radii = np.sqrt(np.sum(points * points, axis=1))
    vals[n_pairs:] = 1.0 - radii
    
    # Soft-min weights
    m = np.min(vals)
    z = -beta * (vals - m)
    z = np.clip(z, -500, 500)
    exp_z = np.exp(z)
    s = np.sum(exp_z)
    if s <= 0:
        return np.zeros_like(points)
    weights = exp_z / s  # sum to 1
    
    # Gradient of softmin = -sum_i w_i * grad_i
    # For pairwise (i,j): value = |c_i - c_j|/2, grad w.r.t. c_i = (c_i - c_j)/|c_i - c_j| * 1/2
    # For boundary i: value = 1 - |c_i|, grad w.r.t. c_i = -c_i/|c_i|
    
    grad = np.zeros_like(points)
    
    # Pairwise contributions
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = weights[idx]
            if w > 1e-12:
                d = dist[i, j]
                if d > 1e-12:
                    grad[i] += w * (points[i] - points[j]) / d * 0.5
                    grad[j] += w * (points[j] - points[i]) / d * 0.5
            idx += 1
    
    # Boundary contributions
    for i in range(n):
        w = weights[n_pairs + i]
        if w > 1e-12:
            r = radii[i]
            if r > 1e-12:
                grad[i] += w * (-points[i] / r)
    
    # We want to maximize softmin, so gradient ascent
    return grad


def _initial_hexagonal(N):
    """Generate initial hexagonal packing."""
    # Start with a hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Select N points closest to center for better symmetry
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            return points[idx]
        spacing *= 0.98


def _local_optimize(points, max_iter=500, deadline=None):
    """
    Local optimization using gradient ascent on soft-min objective.
    """
    points = points.copy()
    n = len(points)
    
    # Adaptive step size
    step = 0.01
    best_score = _score(points)
    beta = 100.0  # Start with moderate beta
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Increase beta over iterations to sharpen
        beta = min(10000.0, 100.0 * (1 + it / 50.0))
        
        grad = _grad_softmin(points, beta)
        
        # Gradient ascent with line search
        # Try a few step sizes
        improved = False
        for trial_step in [step, step * 0.5, step * 0.25, step * 2.0]:
            new_points = points + trial_step * grad
            # Project inside unit circle
            radii = np.sqrt(np.sum(new_points * new_points, axis=1))
            mask = radii > 1.0
            if np.any(mask):
                new_points[mask] /= radii[mask][:, None]
            
            new_score = _score(new_points)
            if new_score > best_score:
                points = new_points
                best_score = new_score
                improved = True
                step = trial_step * 1.1  # Slightly increase
                break
            else:
                step = trial_step * 0.5
        
        if not improved:
            # Reduce step and beta
            step *= 0.5
            if step < 1e-8:
                break
    
    return points, best_score


def construct():
    """
    Construct 300 equal circles inside a unit circle.
    Returns array of shape (N, 2) with centers.
    """
    deadline = time.time() + 110.0  # Safety margin
    
    # Initial hexagonal packing
    points = _initial_hexagonal(N)
    
    # Local optimization with soft-min gradient ascent
    points, score = _local_optimize(points, max_iter=200, deadline=deadline)
    
    # Try a few restarts with different initial perturbations if time permits
    best_points = points
    best_score = score
    
    for restart in range(3):
        if time.time() > deadline - 20:
            break
        
        # Perturb the initial points
        np.random.seed(42 + restart)
        perturbed = _initial_hexagonal(N)
        noise_scale = 0.01 * (restart + 1)
        perturbed = perturbed + np.random.randn(N, 2) * noise_scale
        # Project inside
        radii = np.sqrt(np.sum(perturbed * perturbed, axis=1))
        mask = radii > 1.0
        if np.any(mask):
            perturbed[mask] /= radii[mask][:, None]
        
        points_r, score_r = _local_optimize(perturbed, max_iter=100, deadline=deadline)
        
        if score_r > best_score:
            best_points = points_r
            best_score = score_r
    
    return best_points
# EVOLVE-BLOCK-END
