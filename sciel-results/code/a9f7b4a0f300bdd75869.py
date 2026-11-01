"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    Uses KDTree for efficiency.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    # Pairwise distances via KDTree
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return float(min(min_pair, min_wall))

def _violation(centers, r):
    """
    Compute total violation for radius r.
    Violation = sum of max(0, r - d/2) for pairs + sum of max(0, r - wall_dist) for walls.
    Lower is better. Zero means feasible.
    """
    centers = np.asarray(centers)
    n = len(centers)
    
    # Pairwise violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    pair_violation = np.sum(np.maximum(0, r - dists / 2.0))
    
    # Wall violations
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(wall_dists_x, wall_dists_y)
    wall_violation = np.sum(np.maximum(0, r - wall_dists))
    
    return pair_violation + wall_violation

def _grad_violation(centers, r):
    """
    Compute gradient of violation with respect to centers.
    """
    centers = np.asarray(centers)
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise gradients
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only consider upper triangle to avoid double counting
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d / 2.0 < r and d > 1e-12:
                factor = 0.5 / d
                grad[i] -= factor * (centers[i] - centers[j])
                grad[j] += factor * (centers[i] - centers[j])
    
    # Wall gradients
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    
    for i in range(n):
        if wall_dists_x[i] < r:
            grad[i, 0] += np.sign(centers[i, 0])
        if wall_dists_y[i] < r:
            grad[i, 1] += np.sign(centers[i, 1])
    
    return grad

def _soft_min_obj_and_grad(x, tau):
    """
    Soft-min objective and gradient: minimize -soft_min(all distances).
    x: flattened array of 2*N coordinates
    tau: temperature parameter for soft-min
    """
    centers = x.reshape(-1, 2)
    n = len(centers)
    
    # Collect all relevant distances (pairwise and wall)
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([walls, pair_dists[np.triu_indices(n, k=1)]])
    
    # Soft-min: -log(sum(exp(-d/tau))) * tau
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    obj = -soft_min
    
    # Gradient
    weights = np.exp(-shifted / tau) / np.sum(np.exp(-shifted / tau))
    
    grad_x = np.zeros_like(x)
    
    # Wall gradients
    for i in range(n):
        w = weights[i]
        if w < 1e-15:
            continue
        # d/dx_i wall_dist = -sign(x_i) if wall_x is min, etc.
        if wall_dists_x[i] <= wall_dists_y[i]:
            grad_x[2 * i] += w * np.sign(centers[i, 0])
        else:
            grad_x[2 * i + 1] += w * np.sign(centers[i, 1])
    
    # Pairwise gradients
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = weights[n + idx]
            idx += 1
            if w < 1e-15:
                continue
            d = dists[i, j]
            if d > 1e-12:
                factor = w / (2 * d)
                grad_x[2 * i] += factor * (centers[i, 0] - centers[j, 0])
                grad_x[2 * i + 1] += factor * (centers[i, 1] - centers[j, 1])
                grad_x[2 * j] -= factor * (centers[i, 0] - centers[j, 0])
                grad_x[2 * j + 1] -= factor * (centers[i, 1] - centers[j, 1])
    
    return obj, grad_x

def _sa_search(centers, rng, budget_time):
    """
    Simulated annealing to improve packing.
    """
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    T_start = 0.03
    T_end = 1e-7
    max_steps = 1000000
    
    for step in range(max_steps):
        if time.time() - start_time > budget_time:
            break
        
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Random perturbation
        idx = rng.integers(0, N)
        step_size = 0.015 * (1.0 + 0.3 * np.cos(np.pi * step / max_steps))
        delta = rng.normal(0, step_size, 2)
        
        new = current.copy()
        new[idx] += delta
        new = np.clip(new, -0.5, 0.5)
        
        new_r = _compute_radius(new)
        
        if new_r > current_r:
            current = new
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            delta_r = new_r - current_r
            if delta_r > -5 * T:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current = new
                    current_r = new_r
    
    return best, best_r

def _gradient_polish(centers, r_target, max_iter=500):
    """
    Use gradient descent to minimize violation for a target radius.
    """
    x = centers.flatten().copy()
    lr = 0.005
    
    for i in range(max_iter):
        if time.time() > _deadline:
            break
        
        v = _violation(x.reshape(-1, 2), r_target)
        if v < 1e-10:
            break
        
        g = _grad_violation(x.reshape(-1, 2), r_target).flatten()
        x -= lr * g
        x = np.clip(x, -0.5, 0.5)
    
    return x.reshape(-1, 2)

# Global deadline
_deadline = 0.0

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 145.0  # 5s safety margin
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    best_centers = best_centers + rng.normal(0, 0.01, best_centers.shape)
    best_centers = np.clip(best_centers, -0.49, 0.49)
    
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Simulated Annealing
    remaining = _deadline - time.time()
    sa_budget = min(80.0, remaining * 0.6)
    best_centers, best_radius = _sa_search(best_centers, rng, sa_budget)
    
    # Phase 2: Binary search on radius with violation minimization
    remaining = _deadline - time.time()
    if remaining > 30:
        lo = best_radius
        hi = 0.08  # Upper bound
        
        # Try to find better radius via binary search
        for _ in range(15):
            if time.time() > _deadline - 10:
                break
            
            mid = (lo + hi) / 2
            # Try to find feasible configuration for radius mid
            # Use gradient descent to minimize violation
            trial = best_centers.copy()
            trial = _gradient_polish(trial, mid, max_iter=200)
            
            trial_r = _compute_radius(trial)
            if trial_r >= mid - 1e-9:
                lo = mid
                best_centers = trial
                best_radius = trial_r
            else:
                hi = mid
    
    # Phase 3: Final polish with gradient-based optimization using soft-min
    remaining = _deadline - time.time()
    if remaining > 15:
        # Use soft-min objective with L-BFGS-B
        x0 = best_centers.flatten()
        
        def obj(x):
            return _soft_min_obj_and_grad(x, 0.01)[0]
        
        def obj_grad(x):
            return _soft_min_obj_and_grad(x, 0.01)[1]
        
        try:
            result = opt.minimize(
                obj, x0, jac=obj_grad, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < obj(x0):
                polished = result.x.reshape(-1, 2)
                polished = np.clip(polished, -0.5, 0.5)
                polished_r = _compute_radius(polished)
                if polished_r > best_radius:
                    best_centers = polished
                    best_radius = polished_r
        except Exception:
            pass
    
    # Phase 4: Additional SA restart from polished solution
    remaining = _deadline - time.time()
    if remaining > 10:
        rng2 = np.random.default_rng(123)
        sa_budget2 = min(remaining - 5, 20)
        if sa_budget2 > 5:
            best_centers2, best_radius2 = _sa_search(best_centers, rng2, sa_budget2)
            if best_radius2 > best_radius:
                best_centers = best_centers2
                best_radius = best_radius2
    
    # Final check
    if not np.all(np.isfinite(best_centers)):
        return grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
