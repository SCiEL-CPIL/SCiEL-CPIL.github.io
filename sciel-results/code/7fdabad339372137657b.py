"""Rewritten program for packing 50 circles in a unit circle using Simulated Annealing."""
import numpy as np
import time
import math

N = 50
DEADLINE = 118.0  # Safety margin for 120s limit

def _compute_score(centers):
    """
    Compute the packing score for given centers.
    Score = min(min_pair_dist / 2, min_bound_dist)
    """
    n = len(centers)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def _soft_min_objective(centers, beta):
    """
    Smooth surrogate: minimize -soft_min of (pair_dists/2, bound_dists).
    Returns (loss, gradient) where gradient is w.r.t. centers.
    Uses log-sum-exp for numerical stability.
    """
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Values: pair_dists / 2 for i<j, and (1 - |c_i|) for each i
    # We want to MAXIMIZE the min, so minimize -soft_min
    # soft_min(x_1,...,x_m) = -(1/beta) * log(sum(exp(-beta*x_i)))
    # d(soft_min)/dx_i = -exp(-beta*x_i) / sum(exp(-beta*x_j))
    
    # Collect all values
    # Pair values: dists[i,j]/2 for i<j
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu] / 2.0
    # Boundary values
    norms = np.linalg.norm(centers, axis=1)
    bound_vals = 1.0 - norms
    
    # Combine
    all_vals = np.concatenate([pair_vals, bound_vals])
    
    # For numerical stability
    m = -beta * all_vals
    m_max = np.max(m)
    m_shifted = m - m_max
    exp_m = np.exp(m_shifted)
    log_sum = m_max + np.log(np.sum(exp_m))
    # soft_min = -log_sum / beta
    # loss = -soft_min = log_sum / beta
    loss = log_sum / beta
    
    # Weights
    weights = exp_m / np.sum(exp_m)
    pair_weights = weights[:len(pair_vals)]
    bound_weights = weights[len(pair_vals):]
    
    # Gradient of loss w.r.t. centers
    # d(soft_min)/dx_i = -w_i for each component
    # d(loss)/dx_i = -d(soft_min)/dx_i = w_i
    # But we need chain rule for each center coordinate
    
    grad = np.zeros_like(centers)
    
    # Pair contributions
    for k in range(len(pair_vals)):
        i, j = iu[0][k], iu[1][k]
        w = pair_weights[k]
        # d(pair_val_k)/d(centers[i]) = (centers[i] - centers[j]) / dists[i,j]
        # d(pair_val_k)/d(centers[j]) = (centers[j] - centers[i]) / dists[i,j]
        d = dists[i, j]
        if d > 1e-12:
            diff_vec = (centers[i] - centers[j]) / d
            grad[i] += w * diff_vec
            grad[j] -= w * diff_vec
    
    # Boundary contributions
    for i in range(n):
        w = bound_weights[i]
        # d(bound_val_i)/d(centers[i]) = -centers[i] / |centers[i]|
        norm_i = norms[i]
        if norm_i > 1e-12:
            grad[i] -= w * centers[i] / norm_i
    
    return loss, grad

def _get_hex_initialization(n):
    """
    Generate a hexagonal lattice initialization.
    """
    r_est = 0.12
    s = 2 * r_est
    
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
                
    if len(points) == 0:
        return _get_random_initialization(n, seed=42)
        
    points = np.array(points)
    
    if len(points) < n:
        return _get_random_initialization(n, seed=42)
    
    # Sort by distance to center to pick central ones
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    return points[sorted_idx[:n]]

def _get_random_initialization(n, seed=42):
    """
    Generate random initialization inside the unit circle.
    """
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 2))
    for i in range(n):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x*x + y*y <= 0.95**2:
                centers[i] = [x, y]
                break
    return centers

def _get_ring_initialization(n):
    """
    Generate a ring-based initialization.
    """
    centers = []
    centers.append([0.0, 0.0])
    
    # Approximate distribution: 1, 6, 12, 12, 9, 10? 
    # Total 50. 
    rings = [
        (6, 0.25),
        (12, 0.5),
        (12, 0.75),
        (10, 0.9)
    ]
    
    count = 1
    for count_in_ring, radius in rings:
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
        if count >= n:
            break
            
    return np.array(centers[:n])

def _simulated_annealing(centers, start_time, time_budget):
    """
    Simulated Annealing to maximize the packing score.
    """
    current = centers.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    # Initial temperature
    T = 0.01
    # Cooling rate
    alpha = 0.9995
    # Max perturbation
    max_delta = 0.05
    
    rng = np.random.default_rng(12345)
    
    while time.time() - start_time < time_budget:
        # Perturb a random point
        i = rng.integers(0, N)
        
        # Generate a random move
        angle = rng.uniform(0, 2 * np.pi)
        magnitude = rng.uniform(0, max_delta)
        
        dx = magnitude * np.cos(angle)
        dy = magnitude * np.sin(angle)
        
        new_point = current[i] + np.array([dx, dy])
        
        # Check if new point is inside the circle (with small margin)
        if np.linalg.norm(new_point) > 0.999:
            continue
            
        # Apply move
        old_point = current[i].copy()
        current[i] = new_point
        
        # Compute new score
        new_score = _compute_score(current)
        
        # Acceptance criterion
        if new_score > current_score or (rng.random() < math.exp((new_score - current_score) / T)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = current.copy()
        else:
            # Revert move
            current[i] = old_point
            
        # Cool down
        T *= alpha
        
        # If T is too small, restart with a slightly higher T to escape local minima
        if T < 0.001:
            T = 0.005
            max_delta = 0.02
            
    return best, best_score

def _subgradient_polish(centers, start_time, time_budget, beta_start=50.0, beta_end=500.0):
    """
    Subgradient method on the smooth soft-min objective to polish the solution.
    """
    current = centers.copy()
    best = current.copy()
    best_score = _compute_score(current)
    
    n_iter = 0
    total_iter_budget = 20000  # approximate max iterations
    
    rng = np.random.default_rng(777)
    
    while time.time() - start_time < time_budget and n_iter < total_iter_budget:
        # Animate beta
        progress = min(1.0, n_iter / total_iter_budget)
        beta = beta_start * (beta_end / beta_start) ** progress
        
        loss, grad = _soft_min_objective(current, beta)
        
        # Step size decay
        step_size = 0.01 / (1.0 + 0.01 * n_iter)
        
        # Subgradient step (minimize loss = -soft_min)
        candidate = current - step_size * grad
        
        # Project back into unit circle
        norms = np.linalg.norm(candidate, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            candidate[mask] *= 0.999 / norms[mask]
        
        # Check if improvement in actual score
        new_score = _compute_score(candidate)
        
        if new_score > best_score:
            best_score = new_score
            best = candidate.copy()
        
        # Accept if it doesn't hurt too much (allow some exploration)
        current_score = _compute_score(current)
        if new_score >= current_score - 0.0001:
            current = candidate
        else:
            # Revert and try smaller step
            step_size *= 0.5
            candidate = current - step_size * grad
            norms = np.linalg.norm(candidate, axis=1)
            mask = norms > 0.999
            if np.any(mask):
                candidate[mask] *= 0.999 / norms[mask]
            current = candidate
        
        n_iter += 1
    
    return best, best_score

def _nelder_mead_polish(centers, start_time, time_budget):
    """
    Nelder-Mead local optimization for final high-precision polish.
    Optimizes the negative score (minimize -score).
    """
    from scipy.optimize import minimize
    
    def objective(flat_centers):
        c = flat_centers.reshape(-1, 2)
        return -_compute_score(c)
    
    x0 = centers.flatten()
    
    # Use Nelder-Mead with tight tolerances
    try:
        result = minimize(
            objective, x0,
            method='Nelder-Mead',
            options={
                'xatol': 1e-7,
                'fatol': 1e-9,
                'maxiter': 5000,
                'maxfev': 10000
            }
        )
        
        best_centers = result.x.reshape(-1, 2)
        # Project into unit circle
        norms = np.linalg.norm(best_centers, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            best_centers[mask] *= 0.999 / norms[mask]
        
        return best_centers, _compute_score(best_centers)
    except Exception:
        return centers, _compute_score(centers)

def _polish(centers, start_time, time_budget):
    """
    Local optimization to polish the solution.
    """
    current = centers.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    step_size = 0.001
    rng = np.random.default_rng(54321)
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Try moving each point in 4 directions + random
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
                
            # Directions: up, down, left, right, random
            directions = [
                np.array([0, step_size]),
                np.array([0, -step_size]),
                np.array([step_size, 0]),
                np.array([-step_size, 0]),
                rng.normal(0, step_size, 2)
            ]
            
            for d in directions:
                new_point = current[i] + d
                
                if np.linalg.norm(new_point) > 0.999:
                    continue
                    
                old_point = current[i].copy()
                current[i] = new_point
                
                new_score = _compute_score(current)
                
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best = current.copy()
                    improved = True
                else:
                    current[i] = old_point
        
        # Reduce step size if no improvement
        if not improved:
            step_size *= 0.9
            if step_size < 1e-5:
                break
                
    return best, best_score

def construct():
    start_time = time.time()
    best_centers = None
    best_score = 0.0
    
    # Diverse initializations
    initializations = [
        _get_hex_initialization(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=1),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=99),
    ]
    
    # Time budget per initialization for SA
    # We have ~115s total. 5 inits.
    # Let's give 12s to SA, 4s to subgradient polish per init, but stop if we run out of time.
    
    for i, init in enumerate(initializations):
        # Check time
        if time.time() - start_time > DEADLINE - 15:
            break
            
        # Pre-process: ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
            
        # SA Phase
        sa_time_budget = 12.0
        if time.time() - start_time + sa_time_budget > DEADLINE - 10:
            sa_time_budget = max(0, DEADLINE - 10 - (time.time() - start_time))
            
        if sa_time_budget > 0:
            sa_centers, sa_score = _simulated_annealing(init, start_time, sa_time_budget)
            
            # Subgradient Polish Phase (smooth surrogate)
            subgrad_time_budget = 4.0
            if time.time() - start_time + subgrad_time_budget > DEADLINE - 5:
                subgrad_time_budget = max(0, DEADLINE - 5 - (time.time() - start_time))
                
            if subgrad_time_budget > 0:
                subgrad_centers, subgrad_score = _subgradient_polish(sa_centers, start_time, subgrad_time_budget)
            else:
                subgrad_centers, subgrad_score = sa_centers, sa_score
            
            # Traditional Polish Phase
            polish_time_budget = 3.0
            if time.time() - start_time + polish_time_budget > DEADLINE - 3:
                polish_time_budget = max(0, DEADLINE - 3 - (time.time() - start_time))
                
            if polish_time_budget > 0:
                final_centers, final_score = _polish(subgrad_centers, start_time, polish_time_budget)
            else:
                final_centers, final_score = subgrad_centers, subgrad_score
                
            if final_score > best_score:
                best_score = final_score
                best_centers = final_centers.copy()
                
    # Final high-precision Nelder-Mead polish on the absolute best if time remains
    if best_centers is not None and time.time() - start_time < DEADLINE - 5:
        nm_time_budget = DEADLINE - 5 - (time.time() - start_time)
        if nm_time_budget > 2.0:
            nm_centers, nm_score = _nelder_mead_polish(best_centers, start_time, nm_time_budget)
            if nm_score > best_score:
                best_score = nm_score
                best_centers = nm_centers
    
    # Final quick polish if time remains
    if best_centers is not None and time.time() - start_time < DEADLINE - 1:
        final_time_budget = DEADLINE - 1 - (time.time() - start_time)
        if final_time_budget > 0.5:
            best_centers, best_score = _polish(best_centers, start_time, final_time_budget)
            
    if best_centers is None:
        # Fallback
        best_centers = initializations[0]
        
    return best_centers
