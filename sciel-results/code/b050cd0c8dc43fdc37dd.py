"""Fifty equal circles in a unit square - active matter dynamics with polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_radius(points):
    """Compute the maximum common radius for given circle centres."""
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        return 0.0
    # Check all points are within [-0.5, 0.5]^2
    if np.any(np.abs(points) > 0.5 + 1e-12):
        return 0.0
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    # Distance to walls
    min_x = 0.5 - np.max(points[:, 0])
    max_x = 0.5 + np.min(points[:, 0])
    min_y = 0.5 - np.max(points[:, 1])
    max_y = 0.5 + np.min(points[:, 1])
    min_wall = min(min_x, max_x, min_y, max_y)
    return min(min_pair / 2.0, min_wall)

def _neg_radius(points_flat):
    """Objective function: negative of the radius (to minimize)."""
    points = points_flat.reshape(N, 2)
    r = _compute_radius(points)
    return -r

def _hexagonal_init(rng):
    """Generate initial configuration with hexagonal-like arrangement."""
    # Start with a grid and add small perturbations
    m = 7
    n = 7
    # Create a 7x7 grid and pick 50 points
    xs = np.linspace(-0.4, 0.4, m)
    ys = np.linspace(-0.4, 0.4, n)
    grid = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # Offset every other column for hexagonal packing
            offset = 0.0 if i % 2 == 0 else (ys[1] - ys[0]) / 2
            grid.append([x, y + offset])
    grid = np.array(grid[:N])
    # Add small random perturbation
    grid += rng.uniform(-0.01, 0.01, size=grid.shape)
    # Clip to stay within bounds
    grid = np.clip(grid, -0.45, 0.45)
    return grid

def _grid_init(rng):
    """Generate initial configuration with grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    grid += rng.uniform(-0.005, 0.005, size=grid.shape)
    grid = np.clip(grid, -0.45, 0.45)
    return grid

def _random_init(rng):
    """Generate random initial configuration."""
    points = rng.uniform(-0.45, 0.45, size=(N, 2))
    return points

def _active_matter_step(points, velocities, temp, dt, rng):
    """
    Perform one step of active matter / Langevin dynamics.
    Forces:
    1. Repulsion between circles (inverse square law for short range).
    2. Confinement to the square (soft wall potential).
    3. Swirl force to break symmetry.
    4. Noise (Thermal).
    """
    N = len(points)
    forces = np.zeros_like(points)
    
    # 1. Pairwise Repulsion
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero on diagonal
    
    k_repel = 100.0
    eps = 1e-6
    force_mag = k_repel / (dist_sq + eps)
    
    forces += np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
    
    # 2. Wall Confinement
    wall_margin = 0.4
    k_wall = 1000.0
    
    out_x_pos = np.maximum(points[:, 0] - wall_margin, 0)
    out_x_neg = np.maximum(-wall_margin - points[:, 0], 0)
    forces[:, 0] -= k_wall * (out_x_pos - out_x_neg)
    
    out_y_pos = np.maximum(points[:, 1] - wall_margin, 0)
    out_y_neg = np.maximum(-wall_margin - points[:, 1], 0)
    forces[:, 1] -= k_wall * (out_y_pos - out_y_neg)
    
    # 3. Swirl force
    r_vec = points
    r_mag = np.linalg.norm(r_vec, axis=1, keepdims=True) + 1e-6
    perp = np.column_stack([-r_vec[:, 1], r_vec[:, 0]])
    swirl_strength = 0.5
    forces += swirl_strength * perp / r_mag
    
    # 4. Velocity Update (Langevin)
    damping = 0.95
    noise_mag = temp
    noise = rng.normal(0, noise_mag, size=velocities.shape)
    
    velocities = (velocities + forces * dt) * damping + noise
    
    # 5. Position Update
    points = points + velocities * dt
    
    # 6. Hard Constraint Projection
    points = np.clip(points, -0.5, 0.5)
    
    return points, velocities

def _active_matter_optimize(points, deadline, rng):
    """
    Run active matter simulation to find a good configuration.
    """
    best_points = points.copy()
    best_radius = _compute_radius(best_points)
    
    dt = 0.01
    steps = 2000
    initial_temp = 0.05
    final_temp = 0.001
    
    velocities = np.zeros_like(points)
    
    start_time = time.time()
    
    for i in range(steps):
        if time.time() > deadline:
            break
        
        temp = initial_temp * (final_temp / initial_temp) ** (i / steps)
        
        points, velocities = _active_matter_step(points, velocities, temp, dt, rng)
        
        current_radius = _compute_radius(points)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    return best_points, best_radius

def _local_polish(points, deadline, rng):
    """Run local optimization from given initial points."""
    best_points = points.copy()
    best_radius = _compute_radius(best_points)
    
    flat = best_points.flatten()
    
    for attempt in range(2):
        if time.time() > deadline:
            break
        
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                _neg_radius,
                flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
        except Exception:
            continue
        
        if result.success or result.fun < -best_radius:
            new_points = result.x.reshape(N, 2)
            new_radius = _compute_radius(new_points)
            if new_radius > best_radius:
                best_radius = new_radius
                best_points = new_points
                flat = result.x
    
    return best_points, best_radius

def construct():
    """Construct the arrangement of 50 circles."""
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = 0.0
    
    inits = []
    
    inits.append(_grid_init(rng))
    inits.append(_hexagonal_init(rng))
    
    for _ in range(3):
        inits.append(_random_init(rng))
    
    for init_points in inits:
        if time.time() > deadline:
            break
        
        am_points, am_radius = _active_matter_optimize(init_points, deadline, rng)
        
        if time.time() < deadline:
            points, radius = _local_polish(am_points, deadline, rng)
        else:
            points, radius = am_points, am_radius
        
        if radius > best_radius:
            best_radius = radius
            best_points = points
    
    if best_points is not None and time.time() < deadline - 5:
        flat = best_points.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        try:
            result = minimize(
                _neg_radius,
                flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.success:
                new_points = result.x.reshape(N, 2)
                new_radius = _compute_radius(new_points)
                if new_radius > best_radius:
                    best_points = new_points
                    best_radius = new_radius
        except Exception:
            pass
    
    if best_points is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_points = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points
