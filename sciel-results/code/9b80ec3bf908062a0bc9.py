"""Simulated-annealing approach for packing 50 equal circles in a unit square."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 50
SEED = 42
DEADLINE_FRACTION = 0.85  # Use 85% of the 150s budget as safety margin


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    if min_boundary <= 0:
        return 0.0
    # Use KD-tree for fast nearest-neighbor distances
    tree = cKDTree(centers)
    # Query each point's nearest neighbor (excluding self)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    return min(min_boundary, min_pair)


def _fast_radius(centers):
    """Fast radius computation using KD-tree (no validity checks)."""
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    return min(min_boundary, min_pair)


def _hexagonal_initializer(n_rows=None, seed=None):
    """Generate hexagonal lattice initial configurations with varied parameters."""
    if n_rows is None:
        n_rows = 7  # ~7-8 rows for 50 circles
    rng = np.random.RandomState(seed if seed is not None else SEED)
    
    best_config = None
    best_r = 0.0
    
    for attempt in range(30):
        # Vary number of rows
        nr = rng.randint(5, 11)
        base_per_row = N // nr
        extra = N % nr
        row_counts = [base_per_row + (1 if i < extra else 0) for i in range(nr)]
        
        # Hexagonal spacing parameters
        # Horizontal spacing within a row
        h_spacing = 0.9 / max(max(row_counts) - 1, 1)
        # Vertical spacing between rows
        v_spacing = 0.9 / max(nr - 1, 1)
        # Hex offset for alternating rows
        hex_offset = h_spacing * 0.5 * (0.8 + 0.4 * rng.random())
        
        centers = []
        for row_idx, count in enumerate(row_counts):
            y = -0.45 + 0.9 * row_idx / max(nr - 1, 1)
            if count == 1:
                x = 0.0
                centers.append([x, y])
            else:
                # Center the row
                total_width = (count - 1) * h_spacing
                x_start = -total_width / 2.0
                # Add hexagonal offset for odd rows
                row_offset = hex_offset if row_idx % 2 == 1 else 0.0
                for col_idx in range(count):
                    x = x_start + col_idx * h_spacing + row_offset
                    centers.append([x, y])
        
        centers = np.array(centers[:N])
        # Clip to bounds
        np.clip(centers, -0.49, 0.49, out=centers)
        r = _fast_radius(centers)
        if r > best_r:
            best_r = r
            best_config = centers.copy()
    
    return best_config if best_config is not None else _initial_grid()


def _initial_grid():
    """Generate initial grid configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(float)


def _random_initializer(n_points=N, seed=None):
    """Generate random initial positions within the square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n_points, 2))


def _simulated_annealing(centers, t_start=0.05, t_end=0.0001, n_iterations=None,
                         deadline=None, rng=None, cooling=None):
    """Simulated annealing to maximize the packing radius.
    
    Uses Metropolis acceptance: accept worse moves with probability exp((r_new - r_old)/T).
    """
    if rng is None:
        rng = np.random.RandomState(SEED)
    
    c = centers.copy().astype(float)
    r_current = _fast_radius(c)
    r_best = r_current
    c_best = c.copy()
    
    if n_iterations is None:
        n_iterations = 50000
    
    if cooling is None:
        cooling = (t_end / t_start) ** (1.0 / n_iterations)
    
    for it in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point
        idx = rng.randint(N)
        orig = c[idx].copy()
        
        # Generate a perturbation (Gaussian step)
        step_scale = 0.02 * (t_current / t_start) ** 0.3 if it > 0 else 0.02
        # Actually use temperature-dependent step
        temp_ratio = t_current / t_start
        step = 0.03 * max(0.05, temp_ratio ** 0.5)
        perturbation = rng.normal(0, step, size=2)
        
        c[idx] = orig + perturbation
        # Clip to bounds
        np.clip(c[idx], -0.4999, 0.4999, out=c[idx])
        
        r_new = _fast_radius(c)
        
        # Metropolis acceptance
        delta = r_new - r_current
        if delta >= 0:
            r_current = r_new
            if r_new > r_best:
                r_best = r_new
                c_best = c.copy()
        else:
            # Accept with probability exp(delta / T)
            if t_current > 1e-15:
                accept_prob = np.exp(delta / t_current)
                if rng.random() < accept_prob:
                    r_current = r_new
                else:
                    c[idx] = orig
        
        # Cool down
        t_current *= cooling
    
    return c_best, r_best


def _simulated_annealing_v2(centers, deadline=None, rng=None):
    """Multi-phase simulated annealing with adaptive parameters."""
    if rng is None:
        rng = np.random.RandomState(SEED)
    
    c = centers.copy().astype(float)
    r_current = _fast_radius(c)
    r_best = r_current
    c_best = c.copy()
    
    # Phase 1: High temperature, large steps (exploration)
    t_start = 0.08
    t_end = 0.005
    n_iter_1 = 20000
    cooling_1 = (t_end / t_start) ** (1.0 / n_iter_1)
    t_current = t_start
    
    for it in range(n_iter_1):
        if deadline is not None and time.time() > deadline:
            return c_best, r_best
        
        idx = rng.randint(N)
        orig = c[idx].copy()
        
        step = 0.04 * max(0.1, (t_current / t_start) ** 0.5)
        perturbation = rng.normal(0, step, size=2)
        c[idx] = orig + perturbation
        np.clip(c[idx], -0.4999, 0.4999, out=c[idx])
        
        r_new = _fast_radius(c)
        delta = r_new - r_current
        
        if delta >= 0:
            r_current = r_new
            if r_new > r_best:
                r_best = r_new
                c_best = c.copy()
        else:
            if t_current > 1e-15:
                if rng.random() < np.exp(delta / t_current):
                    r_current = r_new
                else:
                    c[idx] = orig
        
        t_current *= cooling_1
    
    # Phase 2: Medium temperature, medium steps (exploitation)
    t_start_2 = 0.005
    t_end_2 = 0.0003
    n_iter_2 = 30000
    cooling_2 = (t_end_2 / t_start_2) ** (1.0 / n_iter_2)
    t_current = t_start_2
    
    for it in range(n_iter_2):
        if deadline is not None and time.time() > deadline:
            return c_best, r_best
        
        idx = rng.randint(N)
        orig = c[idx].copy()
        
        step = 0.015 * max(0.1, (t_current / t_start_2) ** 0.5)
        perturbation = rng.normal(0, step, size=2)
        c[idx] = orig + perturbation
        np.clip(c[idx], -0.4999, 0.4999, out=c[idx])
        
        r_new = _fast_radius(c)
        delta = r_new - r_current
        
        if delta >= 0:
            r_current = r_new
            if r_new > r_best:
                r_best = r_new
                c_best = c.copy()
        else:
            if t_current > 1e-15:
                if rng.random() < np.exp(delta / t_current):
                    r_current = r_new
                else:
                    c[idx] = orig
        
        t_current *= cooling_2
    
    # Phase 3: Low temperature, small steps (polishing)
    t_start_3 = 0.0003
    t_end_3 = 0.00001
    n_iter_3 = 20000
    cooling_3 = (t_end_3 / t_start_3) ** (1.0 / n_iter_3)
    t_current = t_start_3
    
    for it in range(n_iter_3):
        if deadline is not None and time.time() > deadline:
            return c_best, r_best
        
        idx = rng.randint(N)
        orig = c[idx].copy()
        
        step = 0.005 * max(0.1, (t_current / t_start_3) ** 0.5)
        perturbation = rng.normal(0, step, size=2)
        c[idx] = orig + perturbation
        np.clip(c[idx], -0.4999, 0.4999, out=c[idx])
        
        r_new = _fast_radius(c)
        delta = r_new - r_current
        
        if delta >= 0:
            r_current = r_new
            if r_new > r_best:
                r_best = r_new
                c_best = c.copy()
        else:
            if t_current > 1e-15:
                if rng.random() < np.exp(delta / t_current):
                    r_current = r_new
                else:
                    c[idx] = orig
        
        t_current *= cooling_3
    
    return c_best, r_best


def _force_relaxation(centers, max_iters=200, deadline=None):
    """Quick force relaxation to remove obvious overlaps."""
    c = centers.copy().astype(float)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsion for close pairs
        force_mag = np.where(dists < 0.12, 0.3 / (dists + 0.005), 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_diff = diff / dists[:, :, np.newaxis]
        unit_diff = np.where(np.isfinite(unit_diff), unit_diff, 0.0)
        
        forces = np.sum(force_mag[:, :, np.newaxis] * unit_diff, axis=1)
        
        # Boundary forces
        for axis in range(2):
            dist_to_wall = 0.5 - np.abs(c[:, axis])
            mask = dist_to_wall < 0.08
            forces[mask, axis] -= np.sign(c[mask, axis]) * 0.5 / (dist_to_wall[mask] + 0.005)
        
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 0:
            step = min(0.01, 0.3 / max_force)
            c += step * forces
        
        np.clip(c, -0.4999, 0.4999, out=c)
    
    return c


def construct():
    """Main function: multi-start simulated annealing for circle packing."""
    start_time = time.time()
    total_time_budget = 150.0
    deadline = start_time + total_time_budget * DEADLINE_FRACTION
    
    best_centers = None
    best_radius = 0.0
    
    # Generate diverse initial configurations
    initializers = []
    
    # Hexagonal configurations with different seeds
    for s in range(5):
        init = _hexagonal_initializer(seed=SEED + s)
        initializers.append(init)
    
    # Grid
    initializers.append(_initial_grid())
    
    # Random
    for s in range(3):
        initializers.append(_random_initializer(seed=SEED + 100 + s))
    
    # Evaluate all initializers
    for init_centers in initializers:
        r = _compute_radius(init_centers)
        if r > best_radius:
            best_radius = r
            best_centers = init_centers.copy()
    
    # Force relaxation on each initializer (quick cleanup)
    for i, init_centers in enumerate(initializers):
        if time.time() > deadline:
            break
        if deadline - time.time() < 5.0:
            break
        
        relaxed = _force_relaxation(init_centers, max_iters=100, deadline=deadline)
        r = _fast_radius(relaxed)
        if r > best_radius:
            best_radius = r
            best_centers = relaxed.copy()
    
    # Run SA from the best configuration
    if time.time() < deadline - 10.0:
        rng = np.random.RandomState(SEED)
        sa_result, sa_r = _simulated_annealing_v2(best_centers, deadline=deadline, rng=rng)
        if sa_r > best_radius:
            best_radius = sa_r
            best_centers = sa_result.copy()
    
    # Run SA from a few other good initial configurations
    for i in range(1, min(4, len(initializers))):
        if time.time() > deadline - 15.0:
            break
        if deadline - time.time() < 15.0:
            break
        
        # Use the relaxed version
        relaxed = _force_relaxation(initializers[i], max_iters=50, deadline=deadline)
        rng = np.random.RandomState(SEED + i * 7)
        sa_result, sa_r = _simulated_annealing_v2(relaxed, deadline=deadline, rng=rng)
        if sa_r > best_radius:
            best_radius = sa_r
            best_centers = sa_result.copy()
    
    # Final SA polish on the best solution with very low temperature
    if time.time() < deadline - 5.0:
        rng = np.random.RandomState(SEED + 999)
        # Short, low-temperature SA for final polish
        c = best_centers.copy().astype(float)
        r_current = _fast_radius(c)
        r_best = r_current
        c_best = c.copy()
        
        t_start = 0.0002
        t_end = 0.000005
        n_iter = 15000
        cooling = (t_end / t_start) ** (1.0 / n_iter)
        t_current = t_start
        
        for it in range(n_iter):
            if time.time() > deadline:
                break
            
            idx = rng.randint(N)
            orig = c[idx].copy()
            
            step = 0.003 * max(0.1, (t_current / t_start) ** 0.5)
            perturbation = rng.normal(0, step, size=2)
            c[idx] = orig + perturbation
            np.clip(c[idx], -0.4999, 0.4999, out=c[idx])
            
            r_new = _fast_radius(c)
            delta = r_new - r_current
            
            if delta >= 0:
                r_current = r_new
                if r_new > r_best:
                    r_best = r_new
                    c_best = c.copy()
            else:
                if t_current > 1e-15:
                    if rng.random() < np.exp(delta / t_current):
                        r_current = r_new
                    else:
                        c[idx] = orig
            
            t_current *= cooling
        
        if r_best > best_radius:
            best_radius = r_best
            best_centers = c_best.copy()
    
    # Final validity check
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0:
        best_centers = _initial_grid()
    
    return best_centers


# EVOLVE-BLOCK-END
