"""Optimized sphere packing in a unit cube."""
import numpy as np
import time

N = 100
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s limit


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) < 2:
        return 0.5
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Cube is [-0.5, 0.5]^3
    wall_dists = np.minimum(0.5 - np.abs(centers), axis=1) if centers.ndim == 2 else None
    if centers.ndim == 2:
        wall_dists = np.min(0.5 - np.abs(centers), axis=1)
        min_wall_dist = np.min(wall_dists)
    else:
        min_wall_dist = 0.5
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _initial_grid():
    """Create a simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0)))
    # We need exactly N points, so use a grid of size m x m x m and take first N
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _objective(x, centers_flat, beta):
    """
    Objective function: sum of violations + penalty for radius.
    x contains centers (3N) and radius (1).
    """
    centers = x[:3 * N].reshape(N, 3)
    r = x[3 * N]
    
    # Pairwise constraints: |ci - cj| >= 2r
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 0)
    
    # Violations: max(0, 2r - dist)^2
    violations = np.maximum(0, 2 * r - dists)
    pair_penalty = np.sum(violations ** 2)
    
    # Wall constraints: 0.5 - |c_ik| >= r
    wall_violations = np.maximum(0, r - (0.5 - np.abs(centers)))
    wall_penalty = np.sum(wall_violations ** 2)
    
    # We want to maximize r, so minimize -r + penalty
    return -r + beta * (pair_penalty + wall_penalty)


def _gradient(x, beta):
    """Gradient of the objective function."""
    centers = x[:3 * N].reshape(N, 3)
    r = x[3 * N]
    
    grad = np.zeros_like(x)
    
    # Pairwise part
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1)  # avoid division by zero
    
    violations = np.maximum(0, 2 * r - dists)
    
    # d/d centers_i of sum over j of max(0, 2r - |ci-cj|)^2
    # = sum_j 2 * max(0, 2r - |ci-cj|) * (-1) * (ci - cj) / |ci-cj|
    # For each i, sum over j != i
    
    pair_grad = np.zeros_like(centers)
    for i in range(N):
        for j in range(N):
            if i != j and violations[i, j] > 0:
                pair_grad[i] += 2 * violations[i, j] * (-1) * (centers[i] - centers[j]) / dists[i, j]
    
    # d/dr of sum over pairs max(0, 2r - dist)^2
    # = sum 2 * max(0, 2r - dist) * 2 = 4 * sum violations
    dr_pair = 4 * np.sum(violations)
    
    # Wall part
    wall_violations = np.maximum(0, r - (0.5 - np.abs(centers)))
    wall_grad = np.zeros_like(centers)
    for i in range(N):
        for k in range(3):
            if wall_violations[i, k] > 0:
                # d/d c_ik of max(0, r - (0.5 - |c_ik|))^2
                # = 2 * max(0, ...) * (-1) * sign(c_ik)
                wall_grad[i, k] += 2 * wall_violations[i, k] * (-1) * np.sign(centers[i, k])
    
    # d/dr of wall penalty
    dr_wall = 2 * np.sum(wall_violations)
    
    grad[:3 * N] = pair_grad.flatten() + beta * wall_grad.flatten()
    grad[3 * N] = -1 + beta * (dr_pair + dr_wall)
    
    return grad


def _lbfgs_optimize(centers, r_init, max_iter=2000, beta=1000):
    """Run L-BFGS-B optimization."""
    from scipy.optimize import minimize
    
    x0 = np.concatenate([centers.flatten(), [r_init]])
    
    def obj(x):
        return _objective(x, None, beta)
    
    def grad(x):
        return _gradient(x, beta)
    
    bounds = [(-0.5, 0.5)] * (3 * N) + [(0, 0.5)]
    
    result = minimize(obj, x0, method='L-BFGS-B', jac=grad, bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    new_centers = result.x[:3 * N].reshape(N, 3)
    new_r = result.x[3 * N]
    
    # Clip to valid range
    new_centers = np.clip(new_centers, -0.5, 0.5)
    
    return new_centers, new_r


def construct():
    """Main entry point: returns centers of N spheres."""
    global START_TIME
    if 'START_TIME' not in globals():
        START_TIME = time.time()
    
    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)
    
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT - 5:
        return best_centers
    
    # Phase 1: Initial optimization with moderate beta
    try:
        centers, r = _lbfgs_optimize(best_centers, best_radius, max_iter=1000, beta=100)
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_centers = centers
            best_radius = radius
    except Exception:
        pass
    
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT - 10:
        return best_centers
    
    # Phase 2: Refine with higher beta
    try:
        centers, r = _lbfgs_optimize(best_centers, best_radius, max_iter=2000, beta=1000)
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_centers = centers
            best_radius = radius
    except Exception:
        pass
    
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT - 15:
        return best_centers
    
    # Phase 3: Final polish with very high beta
    try:
        centers, r = _lbfgs_optimize(best_centers, best_radius, max_iter=3000, beta=10000)
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_centers = centers
            best_radius = radius
    except Exception:
        pass
    
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT - 20:
        return best_centers
    
    # Phase 4: Try a few more iterations if time permits
    for beta in [20000, 50000]:
        elapsed = time.time() - START_TIME
        if elapsed > TIME_LIMIT - 10:
            break
        try:
            centers, r = _lbfgs_optimize(best_centers, best_radius, max_iter=2000, beta=beta)
            radius = _compute_radius(centers)
            if radius > best_radius:
                best_centers = centers
                best_radius = radius
        except Exception:
            pass
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centers
