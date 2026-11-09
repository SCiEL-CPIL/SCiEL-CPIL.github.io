"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _min_radius(x):
    """Compute the true minimum radius for configuration x (N,3)."""
    x = x.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)

    wall_dists = np.min(np.abs(0.5 - np.abs(x)), axis=1)
    min_wall_dist = np.min(wall_dists)

    return min(min_pair_dist / 2.0, min_wall_dist)


def _soft_min_radius(x, beta):
    """Smooth approximation of -min_radius using log-sum-exp soft-min.
    
    We want to maximize min(dists/2, wall_dists).
    Equivalently, minimize -min(...) = max(-dists/2, -wall_dists).
    
    Soft-max of values v_i: (1/beta) * log(sum(exp(beta * v_i)))
    Here values are: -pair_dists/2 and -wall_dists.
    """
    x = x.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 0)
    
    # We want soft-max of (-dists/2) and (-wall_dists)
    # For stability, work with squared distances and a modified approach
    # Actually, let's just compute the actual distances for the soft-min
    
    dists = np.sqrt(dists_sq + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    wall_dists = np.min(np.abs(0.5 - np.abs(x)), axis=1)
    
    # Collect all constraint violations (negative values we want to maximize)
    # We want to maximize r such that all pair_dists >= 2r and all wall_dists >= r
    # So we want to maximize the minimum of {pair_dists/2, wall_dists}
    # Equivalent to minimizing the maximum of {-pair_dists/2, -wall_dists}
    
    pair_vals = -dists / 2.0  # shape (N, N), diagonal is -inf
    wall_vals = -wall_dists    # shape (N,)
    
    # Flatten pair_vals excluding diagonal
    pair_vals_flat = pair_vals[~np.eye(N, dtype=bool)]
    
    all_vals = np.concatenate([pair_vals_flat, wall_vals])
    
    # Soft-max: (1/beta) * log(sum(exp(beta * v)))
    # For numerical stability, subtract max
    v_max = np.max(all_vals)
    exp_vals = np.exp(beta * (all_vals - v_max))
    soft_max = v_max + (1.0 / beta) * np.log(np.sum(exp_vals))
    
    return soft_max


def _gradient_soft_min(x, beta):
    """Gradient of _soft_min_radius with respect to x."""
    x = x.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e-15)
    
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    wall_dists = np.min(np.abs(0.5 - np.abs(x)), axis=1)
    
    pair_vals = -dists / 2.0
    wall_vals = -wall_dists
    
    pair_vals_flat = pair_vals[~np.eye(N, dtype=bool)]
    all_vals = np.concatenate([pair_vals_flat, wall_vals])
    
    v_max = np.max(all_vals)
    exp_vals = np.exp(beta * (all_vals - v_max))
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Pairwise gradient contributions
    n_pairs = N * (N - 1) // 2
    pair_weights = weights[:n_pairs]
    
    # Reconstruct pairwise gradient
    for idx in range(n_pairs):
        i, j = np.unravel_index(idx, (N, N - 1))
        if i >= j:
            j = j + 1
        else:
            j = j
        # Actually, let's use a different indexing
        pass
    
    # Simpler: compute gradient numerically or use a different approach
    # Let's use finite differences for the gradient since analytical is complex
    
    # Actually, let's just use the analytical gradient properly
    # For pair (i,j): contribution to grad[i] and grad[j]
    # d/dx_i (-|x_i - x_j|/2) = -(x_i - x_j) / (2 * |x_i - x_j|)
    # d/dx_j (-|x_i - x_j|/2) = (x_i - x_j) / (2 * |x_i - x_j|)
    
    # For wall: d/dx_k (-min(|0.5 - |x_k||, ...))
    # This is more complex. Let's use a simpler approach.
    
    # Given complexity, let's use scipy's minimize with numerical gradient
    return None


def _fibonacci_lattice_init(N, seed=42):
    """Generate initial points using Fibonacci lattice for good distribution."""
    rng = np.random.default_rng(seed)
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = np.zeros((N, 3))
    
    for i in range(N):
        # Fibonacci sphere for 2D projection, then add z
        t = i / N
        z = 2 * t - 1
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points[i] = [x, y, z]
    
    # Scale to fit in [-0.5, 0.5]
    points *= 0.45
    
    # Add small random perturbation
    points += rng.normal(0, 0.01, size=(N, 3))
    points = np.clip(points, -0.48, 0.48)
    
    return points


def construct():
    t0 = time.time()
    deadline = 135.0  # safety margin before 150s kill

    best_x = None
    best_score = -np.inf

    # Multiple initialization strategies
    inits = []
    
    # 1. Fibonacci lattice
    inits.append(_fibonacci_lattice_init(N, seed=42))
    
    # 2. Grid-based
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N].copy())
    
    # 3. Random
    rng = np.random.default_rng(123)
    inits.append(rng.uniform(-0.45, 0.45, size=(N, 3)))

    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline:
            break

        x_init = x_init.copy()
        x_init = np.clip(x_init, -0.48, 0.48)

        # Annealing schedule for beta
        betas = [20, 50, 100, 200, 500, 1000]
        
        current_x = x_init.copy()
        
        for beta in betas:
            if time.time() > deadline:
                break

            def objective(x):
                return _soft_min_radius(x, beta)

            result = minimize(
                objective,
                current_x,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
            )

            if result.success or result.fun < objective(current_x):
                current_x = result.x.reshape(N, 3)
                current_x = np.clip(current_x, -0.5, 0.5)

        # Evaluate true score
        true_score = _min_radius(current_x)
        
        if true_score > best_score:
            best_score = true_score
            best_x = current_x.copy()

        # Additional local refinement with tight beta
        if time.time() < deadline - 10:
            def tight_objective(x):
                return _soft_min_radius(x, 2000)

            result = minimize(
                tight_objective,
                current_x,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )

            if result.success:
                refined_x = np.clip(result.x.reshape(N, 3), -0.5, 0.5)
                refined_score = _min_radius(refined_x)
                
                if refined_score > best_score:
                    best_score = refined_score
                    best_x = refined_x.copy()

    if best_x is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_x = grid[:N].copy()

    # Final validity check
    best_x = np.clip(best_x, -0.5, 0.5)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-10:
                best_x[j] += 1e-5 * np.array([1, 0, 0])

    return best_x
# EVOLVE-BLOCK-END
