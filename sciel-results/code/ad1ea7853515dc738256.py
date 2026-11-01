"""Packing 200 equal circles in a unit square: Velocity-Verlet + L-BFGS-B + Active Set Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 200
DEADLINE = 145.0  # seconds; leave margin before 150s kill

def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < -1e-10:
        return 0.0
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0

def _hex_init(jitter_scale=0.0, seed=0):
    """Generate a hexagonal lattice initialization."""
    best_pts = None
    best_r = 0
    rng = np.random.default_rng(seed)
    
    # Try a wider range of spacings
    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.85, s_est * 1.15, 20):
        pts_list = []
        row = 0
        while True:
            y = -0.5 + s/2 + row * s * np.sqrt(3) / 2
            if y > 0.5 - s/2:
                break
            offset = (row % 2) * s / 2
            x = -0.5 + s/2 + offset
            while x <= 0.5 - s/2:
                pts_list.append((x, y))
                x += s
            row += 1
        if len(pts_list) < N:
            continue
        pts = np.array(pts_list[:N])
        if jitter_scale > 0:
            pts += rng.normal(0, jitter_scale, size=pts.shape)
            pts = np.clip(pts, -0.49, 0.49)
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = pts[:N].copy()
    return best_pts

def _force_directed(pts, time_budget):
    """
    Velocity-Verlet rigid-body simulation with adaptive time-stepping and collision damping.
    Uses repulsive potentials and boundary constraints to settle circles into local equilibrium.
    """
    t0 = time.time()
    positions = pts.copy().astype(np.float64)
    velocities = np.zeros_like(positions)
    current_r = _score(positions)
    
    # Physics parameters
    k_repulse = 100.0      # Repulsion strength
    k_boundary = 50.0      # Boundary constraint strength
    damping = 0.98         # Velocity damping per step
    dt = 0.005             # Initial time step
    min_dt = 1e-5          # Minimum time step
    max_velocity = 0.1     # Maximum velocity clamp
    
    # Adaptive time-stepping: reduce dt if forces are large
    while time.time() - t0 < time_budget:
        # Compute forces
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion (inverse square law)
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        
        # Avoid division by zero and cap forces
        mask = dists > 1e-10
        inv_dists_sq = np.zeros_like(dists)
        inv_dists_sq[mask] = 1.0 / (dists[mask] ** 2)
        
        # Force direction: repulsive
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        forces += k_repulse * np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
        
        # Boundary forces: push back if outside [-0.5, 0.5]
        for dim in range(2):
            # Left wall
            too_left = positions[:, dim] < -0.5
            if np.any(too_left):
                penetration = -0.5 - positions[too_left, dim]
                forces[too_left, dim] += k_boundary * penetration
            # Right wall
            too_right = positions[:, dim] > 0.5
            if np.any(too_right):
                penetration = positions[too_right, dim] - 0.5
                forces[too_right, dim] -= k_boundary * penetration
        
        # Adaptive time-stepping: reduce dt if max force is large
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 1.0:
            dt = min(dt, 0.1 / max_force)
        
        # Velocity-Verlet integration
        # Step 1: Update positions using current velocities and half-step forces
        positions_half = positions + velocities * dt + 0.5 * forces * dt * dt
        
        # Compute forces at half-step positions
        forces_half = np.zeros_like(positions_half)
        
        # Pairwise repulsion at half-step
        dists_half = cdist(positions_half, positions_half)
        np.fill_diagonal(dists_half, np.inf)
        mask_half = dists_half > 1e-10
        inv_dists_sq_half = np.zeros_like(dists_half)
        inv_dists_sq_half[mask_half] = 1.0 / (dists_half[mask_half] ** 2)
        diff_half = positions_half[:, np.newaxis, :] - positions_half[np.newaxis, :, :]
        forces_half += k_repulse * np.sum(diff_half * inv_dists_sq_half[:, :, np.newaxis], axis=1)
        
        # Boundary forces at half-step
        for dim in range(2):
            too_left_half = positions_half[:, dim] < -0.5
            if np.any(too_left_half):
                penetration_half = -0.5 - positions_half[too_left_half, dim]
                forces_half[too_left_half, dim] += k_boundary * penetration_half
            too_right_half = positions_half[:, dim] > 0.5
            if np.any(too_right_half):
                penetration_half = positions_half[too_right_half, dim] - 0.5
                forces_half[too_right_half, dim] -= k_boundary * penetration_half
        
        # Step 2: Update velocities using average of old and new forces
        velocities = velocities * damping + 0.5 * (forces + forces_half) * dt
        
        # Clamp velocities
        vel_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        velocities[vel_norms > max_velocity] *= (max_velocity / vel_norms[vel_norms > max_velocity])
        
        # Step 3: Final position update
        positions = positions_half + velocities * dt
        
        # Hard constraint: keep positions in [-0.5, 0.5]
        positions = np.clip(positions, -0.5, 0.5)
        
        # Check if improved
        new_r = _score(positions)
        if new_r > current_r:
            current_r = new_r
            # Slightly increase dt if stable
            dt = min(dt * 1.01, 0.01)
        else:
            # Reduce dt if not improving
            dt = max(dt * 0.95, min_dt)
        
        # Early stopping if dt is too small
        if dt < min_dt:
            break
    
    return positions

def _soft_min_neg(x, alpha):
    """
    Objective function for L-BFGS-B.
    x: flat array of size 2N.
    alpha: sharpness of the soft-min.
    Returns: -soft_min(distances)
    """
    pts = x.reshape(N, 2)
    
    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()
    
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    
    # Soft min: -1/alpha * log(sum(exp(-alpha * d)))
    val = - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * all_dists)))
    
    # We want to maximize min distance, so minimize -min distance.
    return -val

def _optimize_lbfgs(pts, time_budget):
    """
    Run L-BFGS-B optimization on the initial points with adaptive alpha.
    """
    t0 = time.time()
    x0 = pts.flatten()
    
    phases = [
        {"alpha": 20.0, "max_iter": 200, "gtol": 1e-4},
        {"alpha": 100.0, "max_iter": 200, "gtol": 1e-5},
    ]
    
    for phase in phases:
        if time.time() - t0 > time_budget:
            break
            
        alpha = phase["alpha"]
        
        def obj(x):
            return _soft_min_neg(x, alpha)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', 
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': phase["max_iter"], 'gtol': phase["gtol"]})
            x_opt = res.x
            x0 = x_opt
        except Exception:
            pass
            
    return x0.reshape(N, 2)

def _active_set_polish(pts, time_budget):
    """
    High-precision polish: only move points involved in the minimum distance constraints.
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.default_rng(54321)
    step_size = 0.001
    min_step = 1e-6
    
    while time.time() - t0 < time_budget:
        # Compute constraints
        d_x = 0.5 - np.abs(best_pts[:, 0])
        d_y = 0.5 - np.abs(best_pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)
        min_boundary = np.min(boundary_dists)
        
        dist = cdist(best_pts, best_pts)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        
        current_r = min(min_boundary, min_pair / 2.0)
        
        # Identify active set: points with distance to boundary < current_r * 1.01
        # and pairs with distance < 2 * current_r * 1.01
        active_idx = set()
        
        tight_boundary = np.where(boundary_dists < current_r * 1.01)[0]
        active_idx.update(tight_boundary.tolist())
        
        # Find tight pairs
        # Use KDTree for speed
        tree = cKDTree(best_pts)
        pairs, dists = tree.query(best_pts, k=3)
        for i in range(N):
            for j in range(1, 3):
                if dists[i, j] < 2.0 * current_r * 1.01:
                    active_idx.add(i)
                    active_idx.add(int(pairs[i, j]))
        
        if not active_idx:
            break
            
        active_list = list(active_idx)
        n_active = len(active_list)
        
        # Try perturbing active points
        improved = False
        for _ in range(10): # Try a few times per iteration
            perturb = np.zeros((N, 2))
            perturb[active_list] = rng.normal(0, step_size, size=(n_active, 2))
            trial = best_pts + perturb
            trial = np.clip(trial, -0.5, 0.5)
            
            trial_r = _score(trial)
            if trial_r > best_r:
                best_pts = trial
                best_r = trial_r
                improved = True
                break
        
        if improved:
            step_size *= 0.9
        else:
            step_size *= 0.8
            if step_size < min_step:
                break
                
    return best_pts

def construct():
    t_start = time.time()
    
    # 1. Initialization
    # Try a few jittered hex inits
    best_pts = None
    best_r = 0.0
    
    inits = [
        _hex_init(jitter_scale=0.0, seed=0),
        _hex_init(jitter_scale=0.005, seed=1),
        _hex_init(jitter_scale=0.01, seed=2),
    ]
    
    for pts in inits:
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    # 2. Velocity-Verlet Physics Simulation
    # Use rigid-body mechanics to quickly improve the layout
    remaining = DEADLINE - time.time()
    if remaining > 10:
        fd_budget = min(remaining * 0.35, 25.0)
        fd_pts = _force_directed(best_pts, fd_budget)
        fd_r = _score(fd_pts)
        if fd_r > best_r:
            best_r = fd_r
            best_pts = fd_pts.copy()
            
    # 3. L-BFGS-B Polish
    remaining = DEADLINE - time.time()
    if remaining > 15:
        lbfgs_budget = min(remaining * 0.4, 40.0)
        opt_pts = _optimize_lbfgs(best_pts, lbfgs_budget)
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            
    # 4. Active Set Polish
    remaining = DEADLINE - time.time()
    if remaining > 5:
        polish_budget = min(remaining * 0.8, 30.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
            
    # 5. Final L-BFGS-B pass if time remains
    remaining = DEADLINE - time.time()
    if remaining > 10:
        lbfgs_budget = min(remaining * 0.4, 20.0)
        opt_pts = _optimize_lbfgs(best_pts, lbfgs_budget)
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            
    # 6. Short physics simulation to escape local minima before final polish
    remaining = DEADLINE - time.time()
    if remaining > 8:
        physics_budget = min(remaining * 0.3, 10.0)
        phys_pts = _force_directed(best_pts, physics_budget)
        phys_r = _score(phys_pts)
        if phys_r > best_r:
            best_r = phys_r
            best_pts = phys_pts.copy()
            
    # 7. Final Active Set Polish
    remaining = DEADLINE - time.time()
    if remaining > 3:
        polish_budget = min(remaining * 0.9, 12.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
            
    if best_pts is None:
        best_pts = _hex_init()
        
    return best_pts
