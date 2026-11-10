"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(centers):
    """Compute the minimum radius for the given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Vectorized pairwise distance computation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)


def _soft_min_radius(centers, beta=50.0):
    """
    Smooth surrogate for the minimum radius using log-sum-exp soft-min.
    This provides a differentiable objective for gradient-based optimization.
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Soft-min of pairwise distances / 2
    # Use log-sum-exp trick for numerical stability
    d_half = np.sqrt(dists_sq) / 2.0
    d_half = np.where(np.isinf(d_half), 1.0, d_half)  # handle inf for log-sum-exp
    max_d = np.max(d_half)
    log_sum = np.log(np.sum(np.exp(-(d_half - max_d) * beta))) / beta
    soft_min_pair = max_d - log_sum / beta
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    max_w = np.max(wall_dists)
    log_sum_w = np.log(np.sum(np.exp(-(wall_dists - max_w) * beta))) / beta
    soft_min_wall = max_w - log_sum_w / beta
    
    # Combine: soft-min of the two soft-mins
    max_combined = max(soft_min_pair, soft_min_wall)
    log_sum_c = np.log(np.exp(-(soft_min_pair - max_combined) * beta) + 
                       np.exp(-(soft_min_wall - max_combined) * beta)) / beta
    soft_min_combined = max_combined - log_sum_c / beta
    
    return soft_min_combined


def _negative_soft_min(x, beta=50.0):
    """Objective function for minimization (negative of soft-min radius)."""
    centers = x.reshape(N, 3)
    # Penalty for going outside bounds
    penalty = 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        penalty = 10.0 * (np.max(np.abs(centers) - 0.5) + 1e-6)
    
    return -_soft_min_radius(centers, beta) + penalty


def _soft_min_gradient(x, beta=50.0):
    """
    Numerical gradient of the negative soft-min objective.
    """
    centers = x.reshape(N, 3)
    eps = 1e-7
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        
        f_plus = _negative_soft_min(x_plus, beta)
        f_minus = _negative_soft_min(x_minus, beta)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad


def _optimize_smooth(centers, time_budget, beta_start=10.0, beta_end=100.0):
    """
    Optimize using smooth surrogate with increasing beta (annealing-like).
    Uses L-BFGS-B with numerical gradients.
    """
    start_time = time.time()
    current = centers.copy().flatten()
    best = current.copy()
    best_score = _compute_score(current.reshape(N, 3))
    
    # Start with lower beta (smoother), increase over time
    num_phases = 5
    phase_time = time_budget / num_phases
    
    for phase in range(num_phases):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Beta increases from smooth to sharp
        t = phase / max(num_phases - 1, 1)
        beta = beta_start + (beta_end - beta_start) * t
        
        phase_deadline = start_time + (phase + 1) * phase_time
        
        # Use L-BFGS-B with numerical gradient
        try:
            result = minimize(
                _negative_soft_min,
                current,
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-6, 0.5 - 1e-6)] * (N * 3),
                options={'maxiter': 200, 'maxfun': 500, 'disp': False}
            )
            
            new_centers = result.x.reshape(N, 3)
            new_score = _compute_score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best = result.x.copy()
                current = result.x.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() > phase_deadline - 1.0:
            break
    
    return best.reshape(N, 3), best_score


def _local_search(centers, time_budget, step_size=0.01):
    """
    Local search with coordinate descent on the exact objective.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_score = _compute_score(current)
    
    while time.time() - start_time < time_budget:
        improved = False
        
        for i in range(N):
            for coord in range(3):
                # Try moving in both directions
                for direction in [-1, 1]:
                    new_center = current.copy()
                    new_center[i, coord] += direction * step_size
                    new_center = np.clip(new_center, -0.5 + 1e-9, 0.5 - 1e-9)
                    
                    new_score = _compute_score(new_center)
                    if new_score > best_score:
                        best_score = new_score
                        best = new_center.copy()
                        current = new_center.copy()
                        improved = True
        
        if not improved:
            # Reduce step size
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return best, best_score


def _bcc_initial():
    """Generate a BCC-like initial configuration for 50 spheres."""
    # Use a 4x4x4 grid = 64 positions, take 50
    m = 4
    spacing = 1.0 / m
    centers = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = -0.5 + (i + 0.5) * spacing
                y = -0.5 + (j + 0.5) * spacing
                z = -0.5 + (k + 0.5) * spacing
                centers.append([x, y, z])
    
    centers = np.array(centers[:N])
    return centers


def _random_initial(scale=0.1):
    """Generate a random initial configuration."""
    centers = np.random.uniform(-0.5 + scale, 0.5 - scale, size=(N, 3))
    return centers


def construct():
    """Construct optimal sphere packing."""
    start_time = time.time()
    deadline = 115.0  # safety margin before 150s kill
    
    np.random.seed(42)
    
    # Generate initial configuration
    initial = _bcc_initial()
    
    best_centers = initial.copy()
    best_score = _compute_score(initial)
    
    # Phase 1: Smooth optimization from multiple starts
    num_smooth_restarts = 4
    smooth_time_per = min(20.0, (deadline - 30.0) / num_smooth_restarts)
    
    for restart in range(num_smooth_restarts):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 30.0:
            break
        
        if restart == 0:
            centers = initial.copy()
        else:
            # Different random starts
            scale = 0.05 + 0.03 * restart
            centers = _random_initial(scale)
        
        try:
            optimized, score = _optimize_smooth(centers, time_budget=smooth_time_per)
            if score > best_score:
                best_score = score
                best_centers = optimized.copy()
        except Exception:
            pass
    
    # Phase 2: Local search polish on the best result
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 10.0:
        try:
            polished, polished_score = _local_search(best_centers, time_budget=min(remaining - 5.0, 40.0))
            if polished_score > best_score:
                best_score = polished_score
                best_centers = polished.copy()
        except Exception:
            pass
    
    # Phase 3: Final fine-tuning with very small steps
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 5.0:
        try:
            fine, fine_score = _local_search(best_centers, time_budget=min(remaining - 2.0, 10.0), step_size=0.002)
            if fine_score > best_score:
                best_score = fine_score
                best_centers = fine.copy()
        except Exception:
            pass
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers
# EVOLVE-BLOCK-END
