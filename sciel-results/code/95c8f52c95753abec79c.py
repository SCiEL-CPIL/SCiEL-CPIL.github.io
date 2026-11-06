"""Rewritten program: maximize minimum distance for 100 spheres in unit cube [-0.5,0.5]^3."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.distance import cdist

N = 100
CUBE_HALF = 0.5

def _compute_radius(centers):
    """Compute the valid packing radius for given centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(centers < -CUBE_HALF) or np.any(centers > CUBE_HALF):
        return 0.0
    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    # Wall distances
    wall = CUBE_HALF - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    r = min(min_pair / 2.0, min_wall)
    return max(0.0, r)

def _softmin(distances, t=0.01):
    """Compute soft-min of distances using log-sum-exp trick."""
    d = distances - np.max(distances)
    return -t * np.log(np.sum(np.exp(d / t))) + np.max(distances)

def _objective_and_grad(x, t=0.01):
    """
    Objective: -softmin of all constraints (pairwise distances and wall distances).
    We want to maximize the minimum, so minimize the negative soft-min.
    Returns (value, gradient).
    """
    centers = x.reshape(N, 3)
    
    # Compute pairwise squared distances
    # Use broadcasting for efficiency
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Wall distances: for each center, min over coords of (0.5 - |c_k|)
    abs_c = np.abs(centers)
    wall_dists = CUBE_HALF - np.max(abs_c, axis=1)
    
    # All distances (not squared for softmin, but we need actual distances)
    # Pairwise actual distances
    dists_pair = np.sqrt(sq_dists)
    dists_pair = dists_pair[dists_pair < np.inf]
    
    all_dists = np.concatenate([dists_pair, wall_dists])
    
    # Soft-min
    max_d = np.max(all_dists)
    shifted = all_dists - max_d
    exp_terms = np.exp(shifted / t)
    softmin_val = max_d - t * np.log(np.sum(exp_terms))
    
    value = -softmin_val
    
    # Gradient computation
    # d(softmin)/d(all_dists_i) = -exp(all_dists_i/t) / sum(exp(all_dists_j/t))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros_like(x)
    
    # Gradient for pairwise distances
    # softmin = max_d - t * log(sum(exp(d_i/t)))
    # d(softmin)/d(d_k) for pairwise: 
    # If we think of it as: f = -t*log(sum(exp(d_i/t))) + max_d
    # Actually let's use the formula: d(softmin)/d(d_i) = -exp(d_i/t)/sum(exp(d_j/t)) = -w_i
    # But max_d also depends on d_i, so we need to be careful.
    # For simplicity, let's use a finite-difference-free approach with the chain rule.
    # Actually, the soft-min is: f(d) = -t * log(sum_i exp(d_i/t))
    # df/dd_i = -exp(d_i/t) / sum_j exp(d_j/t) = -w_i
    # But we have the +max_d term which makes it not purely that.
    # Let me reconsider: softmin = max_d - t*log(sum(exp((d_i - max_d)/t)))
    # Let g = sum(exp((d_i - max_d)/t))
    # softmin = max_d - t*log(g)
    # d(softmin)/d(d_k) = d(max_d)/d(d_k) - t * (1/g) * d(g)/d(d_k)
    # d(g)/d(d_k) = exp((d_k - max_d)/t) / t * (1 - d(max_d)/d(d_k))
    # This is getting complex. Let's just use numerical gradient for now if needed,
    # or use a simpler approach.
    
    # For efficiency, let's compute gradient numerically only if needed,
    # or use the analytical form. Actually, let's use a simpler smooth objective.
    
    # Alternative: use a power-sum approximation to min
    # f = (sum d_i^p)^{1/p} for large p approximates min
    # This is differentiable. Let's use p=10 or 20.
    
    # Actually, let's just return value and use numerical gradient via scipy.
    # For now, return None for gradient to use numerical.
    return value, None

def _objective(x):
    """Objective function for scipy minimize."""
    val, _ = _objective_and_grad(x, t=0.005)
    return val

def _objective_psum(x, p=8):
    """Power-sum approximation to min distance. More stable."""
    centers = x.reshape(N, 3)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    dists_pair = np.sqrt(sq_dists)
    dists_pair = dists_pair[dists_pair < np.inf]
    
    abs_c = np.abs(centers)
    wall_dists = CUBE_HALF - np.max(abs_c, axis=1)
    
    all_dists = np.concatenate([dists_pair, wall_dists])
    
    # Power-sum: (sum d_i^p)^{1/p} approximates min for large p
    # But this is actually an approximation to the min, and we want to maximize it.
    # So we minimize -(sum d_i^p)^{1/p}
    # For numerical stability, use log
    log_sum = np.log(np.sum(all_dists ** p))
    psum_val = np.exp(log_sum / p)
    
    return -psum_val

def _initial_grid():
    """Create a 5x5x4 grid initial configuration."""
    # 5 * 5 * 4 = 100
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.4, 0.4, 4)
    grid = np.array([(xi, yi, zi) for zi in z for yi in y for xi in x])
    return grid[:N]

def _initial_fcc_like():
    """Create an FCC-like initial configuration."""
    # FCC structure: simple cubic with basis at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # For a cube, let's create a lattice
    n = 4  # 4x4x4 = 64, not enough
    # Let's try 5x5x4 with some perturbation
    rng = np.random.RandomState(123)
    grid = _initial_grid()
    # Add small random perturbations
    grid += rng.normal(0, 0.01, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    return grid

def _initial_random(seed=0):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-0.45, 0.45, (N, 3))
    # Simple repulsion to avoid initial overlaps
    for _ in range(50):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 0.08:
                    if d > 1e-10:
                        dir_vec = (centers[i] - centers[j]) / d
                        centers[i] += 0.003 * dir_vec
                        centers[j] -= 0.003 * dir_vec
                        moved = True
        centers = np.clip(centers, -0.49, 0.49)
        if not moved:
            break
    return centers

def _local_optimize(centers, time_limit=20.0, method='L-BFGS-B', p=8):
    """
    Local optimization to maximize the minimum distance.
    Uses power-sum objective for smoothness.
    """
    start = time.time()
    x0 = centers.flatten()
    
    best_x = x0.copy()
    best_val = _objective_psum(x0, p=p)
    
    try:
        result = minimize(
            _objective_psum,
            x0,
            args=(p,),
            method=method,
            options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if time.time() - start < time_limit:
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
    except Exception:
        pass
    
    # Try Nelder-Mead if we have time
    if time.time() - start < time_limit * 0.6:
        try:
            result2 = minimize(
                _objective_psum,
                best_x,
                args=(p,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result2.fun < best_val:
                best_val = result2.fun
                best_x = result2.x
        except Exception:
            pass
    
    return best_x.reshape(N, 3)

def _polish(centers, time_limit=10.0):
    """
    High-precision polish pass using smaller step sizes and tighter tolerances.
    """
    start = time.time()
    x0 = centers.flatten()
    best_x = x0.copy()
    
    # Use very small p for more accurate min approximation, but this makes it less smooth
    # Use p=16 for a good balance
    p = 16
    
    best_val = _objective_psum(x0, p=p)
    
    try:
        result = minimize(
            _objective_psum,
            x0,
            args=(p,),
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    return best_x.reshape(N, 3)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Generate multiple initial configurations
    inits = []
    
    # Grid
    inits.append(_initial_grid())
    
    # FCC-like
    inits.append(_initial_fcc_like())
    
    # Random starts with different seeds
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 30:
            break
        inits.append(_initial_random(seed))
    
    best_centers = None
    best_radius = 0.0
    
    for idx, init in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        # Local optimization
        optimized = _local_optimize(init, time_limit=min(15.0, deadline - time.time() - 5))
        
        # Polish
        if time.time() < deadline - 5:
            polished = _polish(optimized, time_limit=min(5.0, deadline - time.time()))
        else:
            polished = optimized
        
        # Compute radius
        radius = _compute_radius(polished)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = polished.copy()
    
    # If we didn't find anything, use the grid
    if best_centers is None:
        best_centers = _initial_grid()
        best_radius = _compute_radius(best_centers)
    
    # Final clip to ensure validity
    best_centers = np.clip(best_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    return best_centers
