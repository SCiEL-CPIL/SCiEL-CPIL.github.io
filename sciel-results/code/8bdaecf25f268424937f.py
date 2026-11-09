"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    wall_dist = 0.5 - np.abs(centres)
    min_wall_dist = np.min(wall_dist)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _soft_min_radius(centres, beta=100.0):
    """
    Smooth surrogate for the minimum radius using log-sum-exp.
    Returns a smooth approximation of min(min_pair_dist/2, min_wall_dist).
    """
    n = len(centres)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances
    wall_dist = 0.5 - np.abs(centres)
    
    # Combine all constraints
    # For pairs: dist/2
    # For walls: wall_dist for each coordinate
    pair_mask = ~np.eye(n, dtype=bool)
    pair_dists = dist[pair_mask] / 2.0
    wall_dists = wall_dist.flatten()
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min: -beta * log(sum(exp(-beta * x)))
    # For numerical stability, subtract max
    x_max = np.max(all_dists)
    exp_terms = np.exp(-beta * (all_dists - x_max))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = x_max - log_sum / beta
    
    return soft_min_val


def _soft_min_radius_grad(centres, beta=100.0):
    """
    Compute soft-min radius and its gradient with respect to centres.
    Returns (value, gradient).
    """
    n = len(centres)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances
    wall_dist = 0.5 - np.abs(centres)
    
    # Combine all constraints
    pair_mask = ~np.eye(n, dtype=bool)
    pair_dists = dist[pair_mask] / 2.0
    wall_dists = wall_dist.flatten()
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min computation
    x_max = np.max(all_dists)
    exp_terms = np.exp(-beta * (all_dists - x_max))
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min_val = x_max - log_sum / beta
    
    # Gradient of soft min with respect to each distance
    weights = exp_terms / sum_exp  # These are positive, sum to 1
    
    # Now compute gradient with respect to centres
    grad = np.zeros_like(centres)
    
    # Pairwise part - vectorized without Python loops
    # Reconstruct weight matrix from flat array
    n_pairs = n * (n - 1)
    w_pair_flat = weights[:n_pairs]
    w_pair = w_pair_flat.reshape(n, n)
    
    inv_dist = np.zeros_like(dist)
    mask = dist > 1e-10
    inv_dist[mask] = 1.0 / dist[mask]
    
    # Vectorized: force_from_pairs[i] = -0.5 * sum_j w_pair[i,j] * diff[i,j] * inv_dist[i,j]
    w_diff = w_pair[:, :, np.newaxis] * diff * inv_dist[:, :, np.newaxis]
    pair_grad = -0.5 * np.sum(w_diff, axis=1)
    
    grad += pair_grad
    
    # Wall part
    w_wall = weights[n_pairs:].reshape(n, 3)
    wall_grad = w_wall * np.sign(centres)
    grad += wall_grad
    
    return soft_min_val, grad


def _neg_soft_radius(centres, beta):
    """Negative of soft-min radius for minimization."""
    val, _ = _soft_min_radius_grad(centres, beta)
    return -val


def _neg_soft_radius_grad(centres, beta):
    """Negative of soft-min radius and its gradient for minimization."""
    val, grad = _soft_min_radius_grad(centres, beta)
    return -val, -grad


def _make_fibonacci_lattice(n):
    """
    Create n points using a 3D Fibonacci lattice (golden angle spiral on sphere,
    then project/scale to cube).
    
    Actually, for better packing in a cube, let's use a deterministic 
    low-discrepancy sequence (Sobol-like or Halton) mapped to the cube.
    """
    # Use Halton sequence for 3D low-discrepancy
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        # Map to [-0.45, 0.45] to keep inside cube with margin
        points.append([x * 0.9 - 0.45, y * 0.9 - 0.45, z * 0.9 - 0.45])
    
    return np.array(points)


def _make_greedy_grid(n, grid_size=20):
    """
    Create n points using greedy farthest-point selection on a dense grid.
    This gives a good initial packing with well-separated points.
    """
    # Create a dense grid of candidate points
    ticks = (np.arange(grid_size) + 0.5) / grid_size - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Greedy farthest-point selection
    selected = [0]
    remaining = list(range(1, len(grid)))
    
    while len(selected) < n and remaining:
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            min_d = np.min(np.linalg.norm(grid[selected] - grid[idx], axis=1))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    centres = grid[selected]
    return centres


def _make_fcc_packing():
    """
    Create an FCC-inspired packing for 50 spheres.
    """
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    nx, ny, nz = 3, 3, 4
    points = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for b in basis:
                    p = np.array([ix, iy, iz]) + b
                    p = (p - 1.25) / 2.5
                    points.append(p)
    
    points = np.array(points)
    
    # Select 50 with maximum minimum distance using greedy algorithm
    selected = [0]
    remaining = list(range(1, len(points)))
    
    while len(selected) < N and remaining:
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            min_d = np.min(np.linalg.norm(points[selected] - points[idx], axis=1))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    centres = points[selected]
    
    rng = np.random.RandomState(42)
    centres += rng.normal(0, 0.003, size=centres.shape)
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _make_grid_packing():
    """Create a grid-based initial packing."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    selected = [0]
    remaining = list(range(1, len(grid)))
    
    while len(selected) < N and remaining:
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            min_d = np.min(np.linalg.norm(grid[selected] - grid[idx], axis=1))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    centres = grid[selected]
    
    rng = np.random.RandomState(123)
    centres += rng.normal(0, 0.002, size=centres.shape)
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _make_random_packing(seed):
    """Create a random initial packing."""
    rng = np.random.RandomState(seed)
    centres = rng.uniform(-0.45, 0.45, size=(N, 3))
    return centres


def _optimize_with_lbfgsb(centres, time_limit=30.0):
    """
    Optimize centres using L-BFGS-B with annealed soft-min surrogate.
    """
    start_time = time.time()
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    c = centres.flatten().copy()
    n = len(centres)
    
    # Bounds: each coordinate in [-0.499, 0.499]
    bounds = [(-0.499, 0.499)] * (n * 3)
    
    # Anneal beta from low to high
    beta_schedule = [20, 50, 100, 200, 500, 1000]
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit - 2:
            break
        
        # Re-evaluate current best
        current_c = c.reshape(n, 3)
        current_radius = _compute_radius(current_c)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = current_c.copy()
        
        # Run L-BFGS-B for a few iterations
        def fun(x):
            c = x.reshape(n, 3)
            return _neg_soft_radius(c, beta)
        
        def grad(x):
            c = x.reshape(n, 3)
            _, g = _soft_min_radius_grad(c, beta)
            return -g.flatten()
        
        try:
            result = minimize(
                fun, c, jac=grad, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            c = result.x
        except Exception:
            pass
        
        # Clip and update best
        c = np.clip(c, -0.499, 0.499)
        current_c = c.reshape(n, 3)
        current_radius = _compute_radius(current_c)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = current_c.copy()
    
    return best_centres, best_radius


def _polish(centres, time_limit=10.0):
    """
    High-precision polish pass using gradient-based optimization with
    a high-beta soft-min surrogate, followed by direct force-directed refinement.
    """
    start_time = time.time()
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    n = len(centres)
    c = centres.copy()
    
    # Phase 1: Gradient-based polish with high beta
    beta = 500.0
    c_flat = c.flatten().copy()
    bounds = [(-0.4999, 0.4999)] * (n * 3)
    
    def fun(x):
        cc = x.reshape(n, 3)
        return _neg_soft_radius(cc, beta)
    
    def grad(x):
        cc = x.reshape(n, 3)
        _, g = _soft_min_radius_grad(cc, beta)
        return -g.flatten()
    
    try:
        if time.time() - start_time < time_limit * 0.5:
            result = minimize(
                fun, c_flat, jac=grad, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            c = np.clip(result.x.reshape(n, 3), -0.4999, 0.4999)
            current_radius = _compute_radius(c)
            if current_radius > best_radius:
                best_radius = current_radius
                best_centres = c.copy()
    except Exception:
        pass
    
    # Phase 2: Direct force-directed refinement
    c = best_centres.copy()
    step = 0.0001
    min_step = 1e-13
    patience = 0
    max_patience = 500
    
    while time.time() - start_time < time_limit:
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        dist = np.sqrt(dist_sq)
        np.fill_diagonal(dist, np.inf)
        
        # Only consider close pairs for efficiency
        threshold = 0.12
        mask = (dist < threshold) & (dist > 1e-10)
        
        # Inverse square law repulsion
        inv_d2 = np.zeros_like(dist)
        inv_d2[mask] = 1.0 / (dist[mask] ** 2)
        forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        # Wall constraints
        for k in range(3):
            above = c[:, k] > 0.495
            below = c[:, k] < -0.495
            forces[above, k] -= (c[above, k] - 0.495) * 500.0
            forces[below, k] += (-0.495 - c[below, k]) * 500.0
        
        # Normalize forces
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        force_mags = np.maximum(force_mags, 1e-10)
        forces = forces / force_mags
        
        c += step * forces
        c = np.clip(c, -0.5 + 1e-10, 0.5 - 1e-10)
        
        current_radius = _compute_radius(c)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = c.copy()
            patience = 0
        else:
            patience += 1
            step = max(step * 0.98, min_step)
        
        if patience > max_patience:
            break
    
    return best_centres, best_radius


def construct():
    """Construct centres of N equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety
    
    best_centres = None
    best_radius = 0.0
    
    # Strategy 1: Greedy grid init (best initial separation) with L-BFGS-B
    if time.time() < deadline - 50:
        try:
            init = _make_greedy_grid(N, grid_size=20)
            time_left = min(45, deadline - time.time() - 5)
            centres, radius = _optimize_with_lbfgsb(init, time_limit=time_left)
            if radius > best_radius:
                best_radius = radius
                best_centres = centres
        except Exception:
            pass
    
    # Strategy 2: Fibonacci/Halton lattice init with L-BFGS-B
    if time.time() < deadline - 40 and best_centres is not None:
        try:
            init = _make_fibonacci_lattice(N)
            time_left = min(35, deadline - time.time() - 5)
            centres, radius = _optimize_with_lbfgsb(init, time_limit=time_left)
            if radius > best_radius:
                best_radius = radius
                best_centres = centres
        except Exception:
            pass
    
    # Strategy 3: FCC-inspired packing with L-BFGS-B
    if time.time() < deadline - 30 and best_centres is not None:
        try:
            init = _make_fcc_packing()
            time_left = min(25, deadline - time.time() - 5)
            centres, radius = _optimize_with_lbfgsb(init, time_limit=time_left)
            if radius > best_radius:
                best_radius = radius
                best_centres = centres
        except Exception:
            pass
    
    # Strategy 4: Grid-based packing with L-BFGS-B
    if time.time() < deadline - 20 and best_centres is not None:
        try:
            init = _make_grid_packing()
            time_left = min(15, deadline - time.time() - 5)
            centres, radius = _optimize_with_lbfgsb(init, time_limit=time_left)
            if radius > best_radius:
                best_radius = radius
                best_centres = centres
        except Exception:
            pass
    
    # Strategy 5: Multiple random restarts with short optimizations
    if time.time() < deadline - 15:
        for seed in [1, 7, 42, 99, 123, 456, 789, 1000, 2024, 555]:
            if time.time() > deadline - 10:
                break
            try:
                init = _make_random_packing(seed)
                time_left = deadline - time.time() - 5
                if time_left < 5:
                    break
                centres, radius = _optimize_with_lbfgsb(init, time_limit=min(8, time_left))
                if radius > best_radius:
                    best_radius = radius
                    best_centres = centres
            except Exception:
                continue
    
    # Final high-precision polish
    if best_centres is not None and time.time() < deadline - 1:
        try:
            polish_time = min(20, deadline - time.time() - 0.5)
            polished, pr = _polish(best_centres, time_limit=polish_time)
            if pr > best_radius:
                best_radius = pr
                best_centres = polished
        except Exception:
            pass
    
    # Fallback
    if best_centres is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    # Final validation
    if not np.all(np.isfinite(best_centres)) or np.any(np.abs(best_centres) > 0.5):
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    return best_centres
# EVOLVE-BLOCK-END
