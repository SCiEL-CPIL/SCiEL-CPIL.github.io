"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def _pairwise_distances_sq(points):
    """Compute squared pairwise distances for an (n, d) array."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    return dist_sq

def _exact_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    dist_sq = _pairwise_distances_sq(points)
    n = len(points)
    mask = ~np.eye(n, dtype=bool)
    d_sq = dist_sq[mask]
    if len(d_sq) == 0:
        return float('inf')
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq <= 1e-12:
        return float('inf')
    ratio = max_d_sq / min_d_sq
    return ratio

def _soft_objective(x, temperature=0.1):
    """
    Smooth surrogate: log(soft_max(d^2)) - log(soft_min(d^2)).
    This is differentiable and easier to optimize.
    """
    points = x.reshape(-1, D)
    dist_sq = _pairwise_distances_sq(points)
    n = len(points)
    mask = ~np.eye(n, dtype=bool)
    d_sq = dist_sq[mask]
    
    if len(d_sq) == 0:
        return 1e6
    
    # Ensure positive distances
    d_sq = np.maximum(d_sq, 1e-12)
    
    # Soft max
    d_max = np.max(d_sq)
    exp_d = np.exp((d_sq - d_max) / temperature)
    soft_max_val = d_max + temperature * np.log(np.sum(exp_d))
    
    # Soft min
    d_min = np.min(d_sq)
    exp_neg_d = np.exp(-(d_sq - d_min) / temperature)
    soft_min_val = d_min - temperature * np.log(np.sum(exp_neg_d))
    
    # Log ratio
    if soft_max_val <= 0 or soft_min_val <= 0:
        return 1e6
    
    obj = np.log(soft_max_val) - np.log(soft_min_val)
    return obj

def _soft_gradient(x, temperature=0.1):
    """
    Analytic gradient of the soft objective.
    """
    points = x.reshape(-1, D)
    n = len(points)
    dist_sq = _pairwise_distances_sq(points)
    mask = ~np.eye(n, dtype=bool)
    d_sq = dist_sq[mask]
    
    if len(d_sq) == 0:
        return np.zeros_like(x)
    
    d_sq = np.maximum(d_sq, 1e-12)
    
    # Soft max weights
    d_max = np.max(d_sq)
    exp_d = np.exp((d_sq - d_max) / temperature)
    w_max = exp_d / np.sum(exp_d)
    soft_max_val = d_max + temperature * np.log(np.sum(exp_d))
    
    # Soft min weights
    d_min = np.min(d_sq)
    exp_neg_d = np.exp(-(d_sq - d_min) / temperature)
    w_min = exp_neg_d / np.sum(exp_neg_d)
    soft_min_val = d_min - temperature * np.log(np.sum(exp_neg_d))
    
    # Gradient of log(soft_max) w.r.t. d_sq[i] = w_max[i] / soft_max_val
    grad_d_max = w_max / soft_max_val
    
    # Gradient of -log(soft_min) w.r.t. d_sq[i] = -w_min[i] / soft_min_val
    grad_d_min = -w_min / soft_min_val
    
    # Total gradient w.r.t. d_sq
    grad_d_sq = grad_d_max + grad_d_min
    
    # Now backpropagate to points
    # d_sq[i,j] = sum_k (points[i,k] - points[j,k])^2
    # d(d_sq[i,j])/d(points[p,q]) = 2*(points[i,q]-points[j,q]) if p==i, -2*(points[i,q]-points[j,q]) if p==j, 0 otherwise
    
    grad_points = np.zeros_like(points)
    n_pairs = len(d_sq)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            g = grad_d_sq[idx]
            idx += 1
            diff = points[i] - points[j]
            grad_points[i] += 2 * g * diff
            grad_points[j] -= 2 * g * diff
    
    return grad_points.flatten()

def _get_initial_candidates(rng):
    """Generate diverse initial candidates."""
    candidates = []
    
    # 1. Regular grid
    side = int(np.ceil(N ** (1 / D)))
    grid = [(i, j) for i in range(side) for j in range(side)][:N]
    candidates.append(np.array(grid, dtype=float).flatten())
    
    # 2. Circular arrangement
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(circle.flatten())
    
    # 3. Hexagonal lattice subset
    hex_points = []
    a = 1.0
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = a * (i + 0.5 * j)
            y = a * (np.sqrt(3) / 2 * j)
            hex_points.append((x, y))
    hex_points = hex_points[:N]
    candidates.append(np.array(hex_points, dtype=float).flatten())
    
    # 4. Random points in unit square
    candidates.append(rng.uniform(0, 1, size=(N, D)).flatten())
    
    # 5. Perturbed grid
    grid_perturbed = np.array(grid, dtype=float) + rng.normal(0, 0.1, size=(N, D))
    candidates.append(grid_perturbed.flatten())
    
    # 6. Perturbed circle
    circle_perturbed = circle + rng.normal(0, 0.1, size=(N, D))
    candidates.append(circle_perturbed.flatten())
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    rng = np.random.default_rng(42)
    candidates = _get_initial_candidates(rng)
    
    best_x = None
    best_score = float('inf')
    
    # Phase 1: Simulated annealing for global exploration
    for cand_idx, x_init in enumerate(candidates):
        if time.time() > deadline - 30.0:
            break
        
        # Simulated annealing
        x = x_init.copy()
        current_score = _exact_score(x.reshape(-1, D))
        best_local_x = x.copy()
        best_local_score = current_score
        
        T = 0.1
        T_min = 1e-6
        alpha = 0.95
        max_iter = 500
        
        for it in range(max_iter):
            if time.time() > deadline - 25.0:
                break
            
            # Propose a move
            x_new = x + rng.normal(0, T, size=x.shape)
            new_score = _exact_score(x_new.reshape(-1, D))
            
            # Accept if better or with probability exp(-(new-old)/T)
            if new_score < current_score:
                x = x_new
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local_x = x_new.copy()
            else:
                prob = np.exp(-(new_score - current_score) / T)
                if rng.random() < prob:
                    x = x_new
                    current_score = new_score
            
            T *= alpha
            if T < T_min:
                break
        
        if best_local_score < best_score:
            best_score = best_local_score
            best_x = best_local_x
    
    # Phase 2: Local refinement with L-BFGS-B on soft objective
    if best_x is not None:
        if time.time() < deadline - 10.0:
            try:
                res = minimize(
                    lambda x: _soft_objective(x, temperature=0.05),
                    best_x,
                    jac=lambda x: _soft_gradient(x, temperature=0.05),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                new_score = _exact_score(res.x.reshape(-1, D))
                if new_score < best_score:
                    best_score = new_score
                    best_x = res.x
            except Exception:
                pass
    
    # Phase 3: Final polish with Nelder-Mead on exact objective
    if best_x is not None and time.time() < deadline - 5.0:
        try:
            res = minimize(
                lambda x: _exact_score(x.reshape(-1, D)),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            new_score = _exact_score(res.x.reshape(-1, D))
            if new_score < best_score:
                best_score = new_score
                best_x = res.x
        except Exception:
            pass
    
    if best_x is None:
        # Fallback: return a grid
        side = int(np.ceil(N ** (1 / D)))
        grid = [(i, j) for i in range(side) for j in range(side)][:N]
        return np.array(grid, dtype=float)
    
    return best_x.reshape(-1, D)
# EVOLVE-BLOCK-END
