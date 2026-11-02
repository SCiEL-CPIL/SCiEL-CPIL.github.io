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
    
    # Pairwise distances via KDTree - query k=3 to be safe
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return float(min(min_pair, min_wall))

def _violation(centers, r):
    """
    Compute total violation for radius r.
    """
    centers = np.asarray(centers)
    n = len(centers)
    
    # Pairwise violations using KDTree for speed
    tree = cKDTree(centers)
    dists, idxs = tree.query(centers, k=6)  # Check up to 5 nearest neighbors
    pair_violation = 0.0
    for i in range(n):
        for j in range(1, 6):
            idx = idxs[i, j]
            d = dists[i, j]
            if d < 2 * r:
                pair_violation += (2 * r - d) / 2.0
    
    # Wall violations
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(wall_dists_x, wall_dists_y)
    wall_violation = np.sum(np.maximum(0, r - wall_dists))
    
    return pair_violation + wall_violation

def _grad_violation(centers, r):
    """
    Compute gradient of violation with respect to centers.
    Vectorized pairwise computation.
    """
    centers = np.asarray(centers)
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise gradients - vectorized
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only add gradient where d/2 < r
    mask = (dists / 2.0 < r) & (dists > 1e-12)
    factors = np.zeros_like(dists)
    factors[mask] = 0.5 / dists[mask]
    
    # grad_i = -sum_j factors[i,j] * (c_i - c_j) for pairs where violation
    # = sum_j factors[i,j] * (c_j - c_i)
    grad_pair = np.einsum('ij,ijk->ik', factors, -diff)
    grad += grad_pair
    
    # Wall gradients
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    
    mask_x = wall_dists_x < r
    mask_y = wall_dists_y < r
    
    grad[mask_x, 0] += np.sign(centers[mask_x, 0])
    grad[mask_y, 1] += np.sign(centers[mask_y, 1])
    
    return grad

def _soft_min_obj_and_grad(x, tau):
    """
    Soft-min objective and gradient: minimize -soft_min(all distances).
    """
    centers = x.reshape(-1, 2)
    n = len(centers)
    
    # Wall distances
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
    Simulated annealing with adaptive step size.
    """
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    T_start = 0.025
    T_end = 1e-8
    max_steps = 500000
    accept_count = 0
    total_count = 0
    
    for step in range(max_steps):
        if time.time() - start_time > budget_time:
            break
        
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Adaptive step size based on acceptance rate
        if total_count > 100:
            accept_rate = accept_count / total_count
            if accept_rate < 0.1:
                step_size = 0.012
            elif accept_rate > 0.5:
                step_size = 0.02
            else:
                step_size = 0.016
        else:
            step_size = 0.015
        
        # Random perturbation
        idx = rng.integers(0, N)
        delta = rng.normal(0, step_size, 2)
        
        new = current.copy()
        new[idx] += delta
        new = np.clip(new, -0.5, 0.5)
        
        new_r = _compute_radius(new)
        
        total_count += 1
        if new_r > current_r:
            current = new
            current_r = new_r
            accept_count += 1
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
                    accept_count += 1
    
    return best, best_r

def _gradient_polish(centers, r_target, max_iter=300):
    """
    Use gradient descent with adaptive learning rate to minimize violation.
    """
    x = centers.flatten().copy()
    lr = 0.01
    v_prev = _violation(x.reshape(-1, 2), r_target)
    
    for i in range(max_iter):
        if time.time() > _deadline:
            break
        
        v = _violation(x.reshape(-1, 2), r_target)
        if v < 1e-10:
            break
        
        # Adaptive learning rate
        if v > v_prev * 1.05:
            lr *= 0.7
        elif v < v_prev * 0.9:
            lr = min(lr * 1.2, 0.05)
        v_prev = v
        
        g = _grad_violation(x.reshape(-1, 2), r_target).flatten()
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-10:
            break
        x -= lr * g
        x = np.clip(x, -0.5, 0.5)
    
    return x.reshape(-1, 2)

def _hex_init(n):
    """
    Generate hexagonal close-packed initial configuration.
    """
    centers = []
    row = 0
    while len(centers) < n:
        x_offset = (row % 2) * 0.5
        for col in range(10):
            x = -0.45 + col * 0.1 + x_offset * 0.05
            y = -0.45 + row * 0.0866
            if -0.48 <= x <= 0.48 and -0.48 <= y <= 0.48:
                centers.append([x, y])
                if len(centers) >= n:
                    break
        row += 1
    
    return np.array(centers[:n])

# Global deadline
_deadline = 0.0

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 145.0  # 5s safety margin
    
    # Try multiple initial configurations
    rng = np.random.default_rng(42)
    
    # Hexagonal initial
    hex_centers = _hex_init(N)
    hex_centers = hex_centers + rng.normal(0, 0.005, hex_centers.shape)
    hex_centers = np.clip(hex_centers, -0.49, 0.49)
    
    # Grid initial with jitter
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N].copy()
    grid = grid + rng.normal(0, 0.01, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    # Pick better initial
    r_hex = _compute_radius(hex_centers)
    r_grid = _compute_radius(grid)
    
    if r_hex >= r_grid:
        best_centers = hex_centers.copy()
    else:
        best_centers = grid.copy()
    
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Simulated Annealing
    remaining = _deadline - time.time()
    sa_budget = min(60.0, remaining * 0.5)
    best_centers, best_radius = _sa_search(best_centers, rng, sa_budget)
    
    # Phase 2: Progressive soft-min L-BFGS polish
    remaining = _deadline - time.time()
    if remaining > 40:
        x0 = best_centers.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        # Progressive beta: start smooth, tighten
        taus = [0.05, 0.02, 0.008, 0.003]
        current_x = x0.copy()
        
        for tau in taus:
            if time.time() > _deadline - 10:
                break
            
            def obj(x):
                return _soft_min_obj_and_grad(x, tau)[0]
            
            def obj_grad(x):
                return _soft_min_obj_and_grad(x, tau)[1]
            
            try:
                result = opt.minimize(
                    obj, current_x, jac=obj_grad, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if result.fun < obj(current_x):
                    current_x = result.x
                    current_x = np.clip(current_x, -0.5, 0.5)
            except Exception:
                pass
        
        polished = current_x.reshape(-1, 2)
        polished_r = _compute_radius(polished)
        if polished_r > best_radius:
            best_centers = polished
            best_radius = polished_r
    
    # Phase 3: Final SA polish
    remaining = _deadline - time.time()
    if remaining > 15:
        rng2 = np.random.default_rng(123)
        sa_budget2 = min(remaining - 5, 25)
        if sa_budget2 > 5:
            best_centers2, best_radius2 = _sa_search(best_centers, rng2, sa_budget2)
            if best_radius2 > best_radius:
                best_centers = best_centers2
                best_radius = best_radius2
    
    # Phase 4: Final gradient polish
    remaining = _deadline - time.time()
    if remaining > 5:
        trial = best_centers.copy()
        trial = _gradient_polish(trial, best_radius * 1.01, max_iter=100)
        trial_r = _compute_radius(trial)
        if trial_r > best_radius:
            best_centers = trial
            best_radius = trial_r
    
    # Final check
    if not np.all(np.isfinite(best_centers)):
        return _hex_init(N)
        
    return best_centers
# EVOLVE-BLOCK-END
