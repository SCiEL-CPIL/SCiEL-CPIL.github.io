"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _radius_for_centers(centers):
    """Compute the radius: min of half pairwise distances and wall distances."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances: min over each coordinate of (0.5 - |c|)
    wall = 0.5 - np.abs(centers)
    min_wall = np.min(wall)
    if min_wall < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(min_wall, min_pair)


def _get_closest_pairs(centers, k=10):
    """Get the k closest pairs of spheres."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    flat_idx = np.argsort(dist.flatten())
    pairs = []
    for idx in flat_idx[:k]:
        i = idx // N
        j = idx % N
        if i != j:
            pairs.append((i, j, dist[i, j]))
    return pairs


def _push_apart(centers, iterations=200, step_scale=1.0):
    """Iteratively push apart the closest pairs."""
    centers = centers.copy()
    for it in range(iterations):
        # Get the closest pair
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find minimum distance pair
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        min_dist = dist[i, j]
        
        if min_dist < 1e-10:
            # Coincident, add small random perturbation
            rng = np.random.RandomState(it)
            centers[i] += rng.uniform(-0.001, 0.001, size=3)
            continue
        
        # Also check wall constraints
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        
        # Determine effective radius (what we're trying to maximize)
        eff_radius = min(min_dist / 2.0, min_wall)
        
        # If the pair is the bottleneck, push them apart
        if min_dist / 2.0 <= min_wall + 1e-12:
            direction = centers[i] - centers[j]
            direction /= (np.linalg.norm(direction) + 1e-15)
            # Push apart by a small amount
            push = 0.001 * step_scale
            centers[i] += direction * push
            centers[j] -= direction * push
        
        # If wall is the bottleneck, push away from wall
        else:
            # Find which sphere is closest to a wall
            wall_dists = 0.5 - np.abs(centers)
            min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            si, coord = min_wall_idx
            # Push away from that wall
            if centers[si, coord] > 0:
                centers[si, coord] -= 0.001 * step_scale
            else:
                centers[si, coord] += 0.001 * step_scale
        
        # Clip to stay in cube
        centers = np.clip(centers, -0.499, 0.499)
    
    return centers


def _neg_radius_lbfgs(centers_flat):
    """Negative radius for L-BFGS-B minimization."""
    centers = centers_flat.reshape(N, 3)
    r = _radius_for_centers(centers)
    return -r


def _make_grid_initial(seed):
    """Create a jittered grid initialization."""
    rng = np.random.RandomState(seed)
    # Use a 4x4x4 grid = 64 positions, take first 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    # Add small random jitter
    centers += rng.uniform(-0.005, 0.005, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _make_random_initial(seed):
    """Create a random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.49, 0.49, size=(N, 3))


def _optimize_local(centers, time_limit=10.0):
    """Optimize a configuration using push-apart and local refinement."""
    start = time.time()
    current = centers.copy()
    current_radius = _radius_for_centers(current)
    
    # Phase 1: Push-apart refinement
    for scale in [1.0, 0.5, 0.25, 0.1]:
        if time.time() - start > time_limit * 0.5:
            break
        new_centers = _push_apart(current, iterations=100, step_scale=scale)
        new_radius = _radius_for_centers(new_centers)
        if new_radius > current_radius + 1e-12:
            current = new_centers
            current_radius = new_radius
    
    # Phase 2: Local optimization using scipy on the full problem
    if time.time() - start < time_limit * 0.8:
        x0 = current.flatten()
        bounds = [(-0.499, 0.499)] * (N * 3)
        try:
            result = minimize(
                _neg_radius_lbfgs,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
            )
            if result.fun < -current_radius + 1e-12:
                new_centers = result.x.reshape(N, 3)
                new_radius = _radius_for_centers(new_centers)
                if new_radius > current_radius:
                    current = new_centers
                    current_radius = new_radius
        except Exception:
            pass
    
    # Phase 3: Final push-apart polish
    if time.time() - start < time_limit * 0.95:
        new_centers = _push_apart(current, iterations=50, step_scale=0.05)
        new_radius = _radius_for_centers(new_centers)
        if new_radius > current_radius + 1e-12:
            current = new_centers
            current_radius = new_radius
    
    return current, current_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_centers = None
    best_radius = 0.0

    # Generate diverse initializations
    initializations = []
    
    # Grid-based initializations with different jitters
    for seed in [42, 123, 456, 789]:
        initializations.append(_make_grid_initial(seed))
    
    # Random initializations
    for seed in [1, 2, 3, 4]:
        initializations.append(_make_random_initial(seed))
    
    # Structured initializations with different grid sizes
    # 5x5x4 = 100, take 50
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid5 = np.array([(x, y, z) for z in ticks[:4] for y in ticks for x in ticks])
    initializations.append(grid5[:N].copy())
    
    # 4x5x3 = 60, take 50
    m1, m2, m3 = 4, 5, 3
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid_alt = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    initializations.append(grid_alt[:N].copy())

    # Optimize each initialization
    for init_idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 15.0:
            break
        
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        n_remaining = len(initializations) - init_idx
        time_per_init = min(12.0, remaining / (n_remaining + 1))
        
        if time_per_init < 2.0:
            break
        
        try:
            optimized, radius = _optimize_local(init_centers, time_limit=time_per_init)
        except Exception:
            continue
        
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized.copy()

    if best_centers is None:
        # Fallback: simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    # Final clip to ensure validity
    best_centers = np.clip(best_centers, -0.499, 0.499)
    return best_centers
# EVOLVE-BLOCK-END
