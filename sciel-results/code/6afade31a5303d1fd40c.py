"""Packing 50 equal spheres in a unit cube via repulsion annealing + Lloyd equalization."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.spatial import Voronoi
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    if not np.all(np.isfinite(centers)):
        return -1.0
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    if np.any(wall_dists < 0):
        return -1.0
    # Pair distances via KDTree for speed
    tree = cKDTree(centers)
    pairs = tree.query_pairs(1.0)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        # Vectorized distance computation for pairs
        idx_i = np.array([p[0] for p in pairs])
        idx_j = np.array([p[1] for p in pairs])
        diffs = centers[idx_i] - centers[idx_j]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        min_pair_dist = np.min(dists)
    return min(np.min(wall_dists), min_pair_dist / 2.0)

def _compute_radius_full(centers):
    """Full O(N^2) computation for accuracy."""
    if not np.all(np.isfinite(centers)):
        return -1.0
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    if np.any(wall_dists < 0):
        return -1.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(np.min(wall_dists), min_pair_dist / 2.0)

def _initial_grid():
    """4x4x4 grid, select 50."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.default_rng(42)
    indices = rng.choice(len(grid), size=N, replace=False)
    return grid[indices].copy()

def _initial_fcc_like():
    """FCC-inspired initial configuration."""
    a = 1.0
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                basis = np.array([
                    [0, 0, 0],
                    [0.5, 0.5, 0],
                    [0.5, 0, 0.5],
                    [0, 0.5, 0.5]
                ]) * a
                for b in basis:
                    pts = np.array([i, j, k]) * a + b
                    points.append(pts)
    points = np.array(points)
    points -= np.mean(points, axis=0)
    max_ext = np.max(np.abs(points))
    points *= (0.45 / max_ext)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(points), size=N, replace=False)
    return points[indices].copy()

def _initial_bcc_like():
    """BCC-inspired initial configuration."""
    a = 1.0
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                basis = np.array([
                    [0, 0, 0],
                    [0.5, 0.5, 0.5]
                ]) * a
                for b in basis:
                    pts = np.array([i, j, k]) * a + b
                    points.append(pts)
    points = np.array(points)
    points -= np.mean(points, axis=0)
    max_ext = np.max(np.abs(points))
    points *= (0.45 / max_ext)
    rng = np.random.default_rng(123)
    indices = rng.choice(len(points), size=N, replace=False)
    return points[indices].copy()

def _initial_random():
    """Random initial configuration."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def _repulsion_force(centers, power=4.0):
    """Compute repulsion forces on each center from all others (inverse-distance^power)."""
    forces = np.zeros_like(centers)
    for i in range(N):
        for j in range(i + 1, N):
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                # Random small perturbation
                diff = np.random.randn(3) * 0.001
                d = np.linalg.norm(diff)
            f = (diff / d) / (d ** power)
            forces[i] += f
            forces[j] -= f
    return forces

def _repulsion_anneal(centers, deadline, t_max=0.02, t_min=0.0005, alpha=0.98,
                       n_steps_per_temp=50, power=4.0):
    """Simulated annealing with repulsion forces to spread spheres apart."""
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    T = t_max
    rng = np.random.default_rng(42)
    
    step_count = 0
    while T > t_min and time.time() < deadline - 15:
        for _ in range(n_steps_per_temp):
            if time.time() > deadline - 15:
                break
            step_count += 1
            # Pick a random sphere
            i = rng.integers(0, N)
            # Compute force on this sphere from all others
            force = np.zeros(3)
            for j in range(N):
                if j == i:
                    continue
                diff = current[i] - current[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    diff = rng.normal(0, 0.001, 3)
                    d = np.linalg.norm(diff)
                force += (diff / d) / (d ** power)
            # Normalize force
            f_norm = np.linalg.norm(force)
            if f_norm > 1e-12:
                force = force / f_norm
            
            # Propose move: repulsion direction + random noise
            step_size = 0.01 * T / t_max + 0.002
            proposed = current.copy()
            proposed[i] += step_size * (force + rng.normal(0, 0.5, 3))
            proposed[i] = np.clip(proposed[i], -0.49, 0.49)
            
            proposed_r = _compute_radius(proposed)
            delta_r = proposed_r - current_r
            
            if delta_r > 0 or rng.random() < np.exp(delta_r / (T + 1e-12)):
                current = proposed
                current_r = proposed_r
                if current_r > best_r:
                    best = current.copy()
                    best_r = current_r
        
        T *= alpha
    
    return best, best_r

def _lloyd_step(centers, n_iters=20, deadline=None):
    """Lloyd's algorithm: move each center toward centroid of its Voronoi cell."""
    current = centers.copy()
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        try:
            vor = Voronoi(current)
        except Exception:
            break
        
        # For each point, find its Voronoi region and compute centroid
        # clipped to the cube [-0.5, 0.5]^3
        new_centers = np.zeros_like(current)
        
        for i in range(N):
            # Get the Voronoi region for point i
            # Find the region index
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0 or -1 in region:
                # Unbounded region - keep original position
                new_centers[i] = current[i]
                continue
            
            vertices = vor.vertices[region]
            
            if len(vertices) == 0:
                new_centers[i] = current[i]
                continue
            
            # Clip vertices to cube
            vertices_clipped = np.clip(vertices, -0.5, 0.5)
            
            # Centroid of clipped vertices (approximation)
            centroid = np.mean(vertices_clipped, axis=0)
            
            # Move toward centroid with damping
            new_centers[i] = 0.7 * current[i] + 0.3 * centroid
            new_centers[i] = np.clip(new_centers[i], -0.49, 0.49)
        
        current = new_centers
    
    return current

def _local_search(centers, deadline, n_rounds=5):
    """Simple local search: try small perturbations of each point."""
    best = centers.copy()
    best_r = _compute_radius(best)
    rng = np.random.default_rng(42)
    
    for rnd in range(n_rounds):
        if time.time() > deadline - 10:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 10:
                break
            # Try random small moves
            for _ in range(10):
                proposed = best.copy()
                proposed[i] += rng.normal(0, 0.005, 3)
                proposed[i] = np.clip(proposed[i], -0.49, 0.49)
                r = _compute_radius(proposed)
                if r > best_r + 1e-12:
                    best = proposed
                    best_r = r
                    improved = True
        if not improved:
            break
    
    return best, best_r

def _polish_lbfgs(centers, deadline):
    """Final polish with L-BFGS-B on a smooth objective."""
    if time.time() > deadline - 15:
        return centers
    
    def objective(x):
        c = x.reshape(N, 3)
        if not np.all(np.isfinite(c)):
            return 1.0
        # Use a soft-min approximation
        wall_dists = 0.5 - np.max(np.abs(c), axis=1)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_half = dists / 2.0
        np.fill_diagonal(pair_half, np.inf)
        
        # Soft min with beta
        beta = 200.0
        M = np.column_stack([wall_dists, pair_half.ravel()])
        # For each point, soft-min over wall and all pairs involving it
        # Simplified: just use the global min with soft approximation
        all_constraints = np.concatenate([wall_dists, pair_half[~np.eye(N, dtype=bool)]])
        x_vec = -beta * all_constraints
        x_max = np.max(x_vec)
        if x_max == -np.inf:
            return 1.0
        lse = np.log(np.sum(np.exp(x_vec - x_max))) + x_max
        r_smooth = -1.0 / beta * lse
        return -r_smooth
    
    try:
        flat = centers.flatten()
        bounds = [(-0.49, 0.49)] * (N * 3)
        res = minimize(
            objective, flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        new_centers = res.x.reshape(N, 3)
        if np.all(np.isfinite(new_centers)):
            return new_centers
    except Exception:
        pass
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Generate multiple initial configurations
    inits = [
        _initial_grid(),
        _initial_fcc_like(),
        _initial_bcc_like(),
        _initial_random()
    ]
    
    best_centers = inits[0].copy()
    best_radius = _compute_radius(best_centers)
    
    for init in inits[1:]:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
    
    # Phase 1: Repulsion annealing from best initial
    if time.time() < deadline - 40:
        sa_centers, sa_r = _repulsion_anneal(
            best_centers, deadline,
            t_max=0.03, t_min=0.0003, alpha=0.97,
            n_steps_per_temp=30, power=3.0
        )
        if sa_r > best_radius:
            best_radius = sa_r
            best_centers = sa_centers.copy()
    
    # Phase 2: Lloyd's algorithm for equalization
    if time.time() < deadline - 25:
        lloyd_centers = _lloyd_step(best_centers, n_iters=15, deadline=deadline)
        r = _compute_radius(lloyd_centers)
        if r > best_radius:
            best_radius = r
            best_centers = lloyd_centers.copy()
    
    # Phase 3: Local search
    if time.time() < deadline - 15:
        ls_centers, ls_r = _local_search(best_centers, deadline, n_rounds=8)
        if ls_r > best_radius:
            best_radius = ls_r
            best_centers = ls_centers.copy()
    
    # Phase 4: L-BFGS polish
    if time.time() < deadline - 10:
        polished = _polish_lbfgs(best_centers, deadline)
        r = _compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
    
    # Phase 5: Another round of Lloyd + local search if time permits
    if time.time() < deadline - 12:
        lloyd_centers = _lloyd_step(best_centers, n_iters=10, deadline=deadline)
        r = _compute_radius(lloyd_centers)
        if r > best_radius:
            best_radius = r
            best_centers = lloyd_centers.copy()
        
        if time.time() < deadline - 8:
            ls_centers, ls_r = _local_search(best_centers, deadline, n_rounds=5)
            if ls_r > best_radius:
                best_radius = ls_r
                best_centers = ls_centers.copy()
    
    # Final validation
    final_radius = _compute_radius_full(best_centers)
    if final_radius < 0 or not np.all(np.isfinite(best_centers)):
        # Fallback
        best_centers = _initial_grid()
        final_radius = _compute_radius_full(best_centers)
    
    return best_centers
