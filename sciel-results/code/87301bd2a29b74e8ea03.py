"""Pack 200 equal circles in a unit square using gradient-based soft-min optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200


def _get_radius(centers):
    """Compute the maximum radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_margin)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r = min(min_dist / 2.0, min_boundary)
    return r


def _soft_min(x, beta):
    """Soft minimum using log-sum-exp approximation."""
    # x is a 1D array of distances
    x_min = np.min(x)
    # Avoid overflow
    x_shifted = x - x_min
    # log-sum-exp: x_min + log(sum(exp(-beta * x_shifted))) / beta
    # For large beta, this approximates min(x)
    max_val = np.max(x_shifted)
    if max_val > 50.0 / beta:
        # Fallback for numerical stability
        return x_min + np.log(np.sum(np.exp(-beta * np.clip(x_shifted, 0, 50.0 / beta)))) / beta
    return x_min + np.log(np.sum(np.exp(-beta * x_shifted))) / beta


def _objective_and_grad(centers_flat, beta):
    """
    Objective: minimize the negative of the soft-min radius.
    Equivalent to maximizing the soft-min radius.
    
    The radius is min(min_pair_dist/2, min_boundary).
    We want to maximize this, so minimize -soft_min_radius.
    
    Actually, let's think about this more carefully.
    The score is r = min(min_dist/2, min_boundary).
    We want to maximize r.
    
    Let's define:
    - d_ij = distance between circle i and j
    - b_i = distance from circle i to nearest wall = 0.5 - max(|x_i|, |y_i|)
    
    We want to maximize min(min_ij(d_ij)/2, min_i(b_i)).
    
    A common approach: minimize the sum of violations or use a barrier.
    But for a smooth objective, we can minimize:
    f = -soft_min(min_ij(d_ij)/2, min_i(b_i))
    
    This is equivalent to maximizing the soft-min of the radius terms.
    
    Let's define the "radius candidates":
    - For each pair (i,j): d_ij / 2
    - For each circle i: b_i
    
    Then r_soft = soft_min(all these candidates, beta)
    We minimize -r_soft.
    
    Gradient: we need d(-r_soft)/d(centers)
    = -d(r_soft)/d(centers)
    
    r_soft = min_k(c_k) where c_k are the candidates
    d(r_soft)/d(c_k) = softmax weights * (1/n_candidates) approximately
    More precisely: r_soft = -1/beta * log(sum(exp(-beta * c_k)))
    d(r_soft)/d(c_k) = exp(-beta * c_k) / sum(exp(-beta * c_j))
    """
    centers = centers_flat.reshape(N, 2)
    
    # Compute all pair distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary margins
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Collect all radius candidates
    # Pair distances / 2
    pair_radii = dists / 2.0
    # Remove diagonal (self)
    pair_radii = pair_radii[~np.eye(N, dtype=bool)]
    
    # All candidates
    all_candidates = np.concatenate([pair_radii, boundary])
    
    # Soft min
    # For numerical stability, shift by min
    c_min = np.min(all_candidates)
    c_shifted = all_candidates - c_min
    
    # Compute softmax weights for the soft min
    # weights_k = exp(-beta * c_k) / sum(exp(-beta * c_j))
    # For soft min: r_soft = c_min + log(sum(exp(-beta * c_shifted))) / beta
    # d(r_soft)/d(c_k) = exp(-beta * c_k) / sum(exp(-beta * c_j))
    
    exp_vals = np.exp(-beta * c_shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # weights for each candidate
    
    r_soft = c_min + np.log(sum_exp) / beta
    
    # Gradient of r_soft with respect to centers
    # d(r_soft)/d(centers) = sum_k weights_k * d(c_k)/d(centers)
    
    grad = np.zeros((N, 2))
    
    # Pair contributions
    # For each pair (i,j), c_ij = d_ij / 2
    # d(c_ij)/d(centers_i) = (centers_i - centers_j) / (2 * d_ij)
    # d(c_ij)/d(centers_j) = (centers_j - centers_i) / (2 * d_ij)
    
    # We need to map weights back to pairs
    # The pair_radii array is in row-major order, skipping diagonal
    # Let's recompute more efficiently
    
    # Reset grad
    grad = np.zeros((N, 2))
    
    # Process pairs
    for i in range(N):
        for j in range(i + 1, N):
            d = dists[i, j]
            if d < 1e-10:
                continue
            # Weight for this pair
            # We need to find the index of this pair in the weights array
            # pairs are stored as (0,1), (0,2), ..., (0,N-1), (1,2), ...
            # Index = i*(N-i-1) + (j - i - 1) for i < j
            idx = i * (N - i - 1) + (j - i - 1)
            w = weights[idx]
            
            # Gradient contribution
            diff_ij = centers[i] - centers[j]
            grad_i = w * diff_ij / (2.0 * d)
            grad[j] -= grad_i  # d(c)/d(centers_j) = -d(c)/d(centers_i)
            grad[i] += grad_i
    
    # Boundary contributions
    # For circle i, c_i = 0.5 - max(|x_i|, |y_i|)
    # d(c_i)/d(x_i) = -sign(x_i) if |x_i| > |y_i|, else 0
    # d(c_i)/d(y_i) = -sign(y_i) if |y_i| >= |x_i|, else 0
    
    for i in range(N):
        w = weights[N * (N - 1) // 2 + i]  # boundary weights start after all pairs
        ax = np.abs(centers[i, 0])
        ay = np.abs(centers[i, 1])
        if ax > ay:
            grad[i, 0] -= w * np.sign(centers[i, 0])
        elif ay > ax:
            grad[i, 1] -= w * np.sign(centers[i, 1])
        else:
            # Tie: split gradient
            grad[i, 0] -= w * 0.5 * np.sign(centers[i, 0])
            grad[i, 1] -= w * 0.5 * np.sign(centers[i, 1])
    
    # We're minimizing -r_soft, so objective = -r_soft, grad = -grad_r_soft
    objective = -r_soft
    grad_flat = -grad.flatten()
    
    return objective, grad_flat


def _generate_hex_initial(n):
    """Generate a hexagonal grid initialization."""
    # Optimal hex packing density is ~0.9069
    # For n points in unit square, area per point ~ 1/n
    # Hex cell: side length s, area = (sqrt(3)/2) * s^2
    # For hex packing of circles with center distance d, cell area = 2*sqrt(3)*d^2/4 = sqrt(3)/2 * d^2
    # So d ~ sqrt(1/n) approximately
    d = np.sqrt(1.0 / n)
    
    # Generate hex grid
    # Hex grid: rows offset by d/2 horizontally
    # Vertical spacing: d * sqrt(3) / 2
    
    points = []
    row = 0
    while len(points) < n:
        y = -0.5 + d * np.sqrt(3) / 2 * (row + 0.5)
        if y > 0.5:
            break
        if row % 2 == 0:
            x_start = -0.5 + d / 2
        else:
            x_start = -0.5 + d
        x = x_start
        while x < 0.5 and len(points) < n:
            points.append([x, y])
            x += d
        row += 1
    
    # If we don't have enough, add random points
    while len(points) < n:
        points.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
    
    points = np.array(points[:n])
    
    # Add small jitter to break symmetry
    points += np.random.normal(0, 0.001, (n, 2))
    points = np.clip(points, -0.49, 0.49)
    
    return points


def _generate_random_initial(n):
    """Generate a random initialization."""
    return np.random.uniform(-0.45, 0.45, (n, 2))


def _polish(centers, deadline):
    """
    Local polish using direct minimization of the exact (non-smooth) objective.
    Uses small perturbations and accepts improvements.
    """
    centers = centers.copy()
    best_score = _get_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.005
    n_iters = 5000
    iters_done = 0
    
    for _ in range(n_iters):
        if time.time() > deadline:
            break
        
        iters_done += 1
        
        # Random perturbation
        mask = np.random.random((N, 2)) < 0.3
        perturbation = np.random.normal(0, step_size, (N, 2)) * mask
        new_centers = centers + perturbation
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_score = _get_radius(new_centers)
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            centers = new_centers.copy()
            # Reduce step size slightly
            step_size *= 0.99
        else:
            step_size *= 0.995
    
    return best_centers, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_centers = None
    best_score = 0.0
    
    # Try multiple initializations and optimization runs
    inits = [
        ("hex", _generate_hex_initial),
        ("random", _generate_random_initial),
    ]
    
    for init_name, init_func in inits:
        if time.time() > deadline - 30:
            break
        
        centers = init_func(N)
        
        # Run L-BFGS-B with increasing beta
        for beta in [20, 100, 500]:
            if time.time() > deadline - 15:
                break
            
            # Flatten centers
            centers_flat = centers.flatten()
            
            # Run optimization
            try:
                res = minimize(
                    _objective_and_grad,
                    centers_flat,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 300,
                        'ftol': 1e-14,
                        'gtol': 1e-8,
                        'maxls': 50
                    }
                )
                centers = res.x.reshape(N, 2)
                centers = np.clip(centers, -0.5, 0.5)
            except Exception:
                continue
        
        # Polish with local search
        if time.time() > deadline - 10:
            break
        
        centers, score = _polish(centers, deadline - 5)
        
        if score > best_score:
            best_score = score
            best_centers = centers
    
    # Final polish on the best result
    if best_centers is not None and time.time() < deadline - 3:
        best_centers, best_score = _polish(best_centers, deadline - 1)
    
    if best_centers is None:
        # Fallback: hex initial
        best_centers = _generate_hex_initial(N)
    
    return best_centers
