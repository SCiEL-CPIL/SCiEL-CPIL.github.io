"""Packing 50 equal circles in a unit square; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
import itertools
import math

# Deadline guard: keep a safety margin before the 150s wall-clock limit.
_DEADLINE = 150.0
_SAFETY = 10.0  # seconds of margin


def _compute_radius(centers):
    """Compute the largest common radius for the given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        raise ValueError("Wrong shape")
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    # Min distance to boundary
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    r_boundary = np.min(boundary)
    if r_boundary <= 0:
        return 0.0
    # Min pairwise distance / 2
    n = len(centers)
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    r_pair = min_pair / 2.0
    return min(r_boundary, r_pair)


def _objective(centers):
    """Objective to minimize: negative of the radius (we maximize radius)."""
    centers = np.asarray(centers, dtype=np.float64).reshape(N, 2)
    # Compute radius
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    r_boundary = np.min(boundary)
    min_pair = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    r_pair = min_pair / 2.0
    r = min(r_boundary, r_pair)
    return -r


def _construct_hexagonal():
    """Create a hexagonal packing pattern for 50 circles."""
    # Hexagonal lattice: rows offset
    # For a square, we can try a hexagonal arrangement
    # Let's create rows with alternating offsets
    centers = []
    # Approximate hexagonal packing
    # Row height = r * sqrt(3)
    # Horizontal spacing = 2r
    # Let's try a grid-like hexagonal pattern
    m = 7  # 7x7 = 49, plus 1 extra
    # Use a hexagonal lattice
    r_est = 0.07
    row_h = r_est * math.sqrt(3)
    col_w = 2 * r_est
    # Start from a base offset
    base_x = -0.5 + r_est
    base_y = -0.5 + r_est
    row = 0
    while len(centers) < N:
        y = base_y + row * row_h
        if y > 0.5 - r_est:
            break
        if row % 2 == 0:
            x_start = base_x
        else:
            x_start = base_x + col_w / 2
        x = x_start
        while x <= 0.5 - r_est and len(centers) < N:
            centers.append([x, y])
            x += col_w
        row += 1
    # Pad if needed
    while len(centers) < N:
        centers.append([0.0, 0.0])
    return np.array(centers[:N], dtype=np.float64)


def _construct_grid():
    """Create a square grid packing."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _construct_random(seed=42):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-0.5 + 0.01, 0.5 - 0.01, size=(N, 2))
    return centers


def _gradient_polish(centers, max_iter=2000, tol=1e-10):
    """Use gradient-based optimization to polish the configuration."""
    x0 = centers.reshape(-1)
    
    def obj(x):
        c = x.reshape(N, 2)
        # Soft min to make differentiable
        # Use a large weight for the min to approximate the true min
        boundary = 0.5 - np.max(np.abs(c), axis=1)
        # Pairwise distances
        dists = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(c[i] - c[j])
                dists.append(d)
        dists = np.array(dists)
        r_pair = np.min(dists) / 2.0
        r_boundary = np.min(boundary)
        r = min(r_pair, r_boundary)
        return -r
    
    def gradient(x):
        c = x.reshape(N, 2)
        boundary = 0.5 - np.max(np.abs(c), axis=1)
        r_boundary = np.min(boundary)
        # Find which boundary constraint is active
        active_boundary_idx = np.argmin(boundary)
        
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(c[i] - c[j])
                dists[i, j] = d
                dists[j, i] = d
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        r_pair = min_dist / 2.0
        
        r = min(r_pair, r_boundary)
        
        grad = np.zeros_like(x)
        if r_pair <= r_boundary:
            # Active constraint is pairwise
            for i in range(N):
                for j in range(i + 1, N):
                    if abs(dists[i, j] - min_dist) < 1e-12:
                        diff = c[i] - c[j]
                        if np.linalg.norm(diff) > 1e-15:
                            diff = diff / np.linalg.norm(diff)
                            grad[2*i] += 0.5 * diff
                            grad[2*i+1] += 0.5 * diff[1]
                            grad[2*j] -= 0.5 * diff
                            grad[2*j+1] -= 0.5 * diff[1]
                        else:
                            grad[2*i] += 0.5
                            grad[2*j] -= 0.5
        else:
            # Active constraint is boundary
            i = active_boundary_idx
            # Gradient of (0.5 - max(|x_i|, |y_i|))
            if abs(c[i, 0]) >= abs(c[i, 1]):
                if c[i, 0] > 0:
                    grad[2*i] = -1
                else:
                    grad[2*i] = 1
            else:
                if c[i, 1] > 0:
                    grad[2*i+1] = -1
                else:
                    grad[2*i+1] = 1
        return -grad  # We're minimizing -r, so gradient of -r
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', jac=gradient, 
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
        if res.success or res.fun < obj(x0):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return centers


def _simulated_annealing(centers, n_iter=5000, t_start=0.05, t_end=0.001, seed=42):
    """Simple simulated annealing to improve the packing."""
    rng = np.random.RandomState(seed)
    best = centers.copy()
    best_r = _compute_radius(best)
    current = centers.copy()
    current_r = best_r
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - _SAFETY:
            break
        # Generate a random perturbation
        idx = rng.randint(N)
        delta = rng.normal(0, t_start * 0.1, size=2)
        new_pos = current[idx].copy() + delta
        # Keep within bounds
        new_pos = np.clip(new_pos, -0.5 + 0.001, 0.5 - 0.001)
        current[idx] = new_pos
        
        new_r = _compute_radius(current)
        
        # Accept or reject
        if new_r >= current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
        else:
            # Metropolis criterion
            delta_r = current_r - new_r
            if rng.uniform() < math.exp(-delta_r / (t_start * 0.01)):
                current_r = new_r
            else:
                current[idx] = current[idx] - delta
        
        # Cool down
        t = t_start * (t_end / t_start) ** (it / n_iter)
    
    return best


def _construct():
    start_time = time.time()
    
    # Generate multiple initial configurations
    inits = []
    
    # 1. Grid
    grid = _construct_grid()
    inits.append(grid)
    
    # 2. Hexagonal
    hexag = _construct_hexagonal()
    inits.append(hexag)
    
    # 3. Several random starts
    for seed in range(5):
        rand = _construct_random(seed=100 + seed)
        inits.append(rand)
    
    # Polish each initial configuration
    best_centers = None
    best_radius = -1
    
    for init in inits:
        if time.time() > _DEADLINE - _SAFETY:
            break
        
        # First, do a quick SA to get a good starting point
        sa_result = _simulated_annealing(init, n_iter=1000, t_start=0.03, t_end=0.005, seed=42)
        
        # Then gradient polish
        polished = _gradient_polish(sa_result, max_iter=500)
        
        r = _compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished
    
    # Do a few more SA iterations from the best
    if time.time() < _DEADLINE - _SAFETY:
        final = _simulated_annealing(best_centers, n_iter=2000, t_start=0.01, t_end=0.001, seed=123)
        r_final = _compute_radius(final)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = final
    
    # Final gradient polish
    if time.time() < _DEADLINE - _SAFETY:
        final_polished = _gradient_polish(best_centers, max_iter=300)
        r_fp = _compute_radius(final_polished)
        if r_fp > best_radius:
            best_radius = r_fp
            best_centers = final_polished
    
    # Ensure validity
    result = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    
    # Final check
    r_check = _compute_radius(result)
    if r_check <= 0:
        # Fallback to grid
        result = _construct_grid()
    
    return result

# EVOLVE-BLOCK-END
