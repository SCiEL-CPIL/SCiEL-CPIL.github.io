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


def _repulsion_optimize(x_init, max_iter=5000, time_limit=100.0, verbose=False):
    """
    Repulsion-based optimization for sphere packing.
    
    At each step:
    1. Find the closest pair and the most boundary-violating sphere.
    2. Move points to relieve the most violated constraint.
    3. Use adaptive step size with backtracking.
    """
    x = x_init.copy().reshape(N, 3)
    x = np.clip(x, -0.49, 0.49)
    
    best_x = x.copy()
    best_score = _min_radius(x)
    
    t0 = time.time()
    
    for it in range(max_iter):
        if time.time() - t0 > time_limit:
            break
        
        # Compute all pairwise distances and wall distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        wall_dists = np.min(np.abs(0.5 - np.abs(x)), axis=1)
        
        # Current minimum radius
        min_pair_dist = np.min(dists)
        min_wall_dist = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Find the most violated constraint
        # We want to increase the minimum of {dists/2, wall_dists}
        # The most violated is the smallest value
        
        # Get indices of closest pair
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx[0], min_pair_idx[1]
        
        # Get index of most boundary-violating sphere
        wall_idx = np.argmin(wall_dists)
        
        # Decide which constraint to relieve
        if min_pair_dist / 2.0 <= min_wall_dist:
            # Relieve closest pair
            direction = x[i] - x[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-15:
                direction = np.random.randn(3)
                norm = np.linalg.norm(direction)
            direction /= norm
            
            # Try to move i and j apart
            step = 0.01
            for _ in range(20):
                x_new = x.copy()
                x_new[i] += step * direction
                x_new[j] -= step * direction
                x_new = np.clip(x_new, -0.49, 0.49)
                
                new_score = _min_radius(x_new)
                if new_score > current_r + 1e-12:
                    x = x_new
                    break
                step *= 0.5
        else:
            # Relieve boundary violation
            # Move the most violating sphere toward center
            direction = -x[wall_idx]  # toward origin
            norm = np.linalg.norm(direction)
            if norm < 1e-15:
                direction = np.array([0.1, 0.1, 0.1])
                norm = np.linalg.norm(direction)
            direction /= norm
            
            step = 0.01
            for _ in range(20):
                x_new = x.copy()
                x_new[wall_idx] += step * direction
                x_new = np.clip(x_new, -0.49, 0.49)
                
                new_score = _min_radius(x_new)
                if new_score > current_r + 1e-12:
                    x = x_new
                    break
                step *= 0.5
        
        # Track best
        current_score = _min_radius(x)
        if current_score > best_score:
            best_score = current_score
            best_x = x.copy()
    
    return best_x, best_score


def _soft_min_radius(x, beta):
    """Smooth approximation of -min_radius using log-sum-exp soft-min."""
    x = x.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e-15)
    
    dists = np.sqrt(dists_sq + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    wall_dists = np.min(np.abs(0.5 - np.abs(x)), axis=1)
    
    pair_vals = -dists / 2.0
    wall_vals = -wall_dists
    
    pair_vals_flat = pair_vals[~np.eye(N, dtype=bool)]
    all_vals = np.concatenate([pair_vals_flat, wall_vals])
    
    v_max = np.max(all_vals)
    exp_vals = np.exp(beta * (all_vals - v_max))
    soft_max = v_max + (1.0 / beta) * np.log(np.sum(exp_vals))
    
    return soft_max


def _fibonacci_lattice_init(N, seed=42):
    """Generate initial points using Fibonacci lattice for good distribution."""
    rng = np.random.default_rng(seed)
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = np.zeros((N, 3))
    
    for i in range(N):
        t = i / N
        z = 2 * t - 1
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points[i] = [x, y, z]
    
    points *= 0.45
    points += rng.normal(0, 0.01, size=(N, 3))
    points = np.clip(points, -0.48, 0.48)
    
    return points


def _grid_init(N):
    """Grid-based initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_init(N, seed=123):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))


def _best_first_grid_init(N):
    """
    Best-first grid initialization: place spheres at grid positions 
    that maximize the minimum distance to already-placed spheres.
    """
    # Create a fine grid of candidate positions
    n_per_dim = 8
    ticks = (np.arange(n_per_dim) + 0.5) / n_per_dim - 0.5
    candidates = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    placed = []
    for _ in range(N):
        if len(candidates) == 0:
            break
        
        if len(placed) == 0:
            # Place first sphere at center
            best_idx = np.argmin(np.sum(candidates ** 2, axis=1))
            placed.append(candidates[best_idx].copy())
            candidates = np.delete(candidates, best_idx, axis=0)
        else:
            # For each candidate, compute min distance to placed spheres
            min_dists = np.min(
                np.sqrt(np.sum((candidates[:, np.newaxis, :] - np.array(placed)[np.newaxis, :, :]) ** 2, axis=2)),
                axis=1
            )
            # Also consider wall distance
            wall_dists = np.min(np.abs(0.5 - np.abs(candidates)), axis=1)
            # Score: min of pair distance and wall distance
            scores = np.minimum(min_dists, wall_dists)
            best_idx = np.argmax(scores)
            placed.append(candidates[best_idx].copy())
            candidates = np.delete(candidates, best_idx, axis=0)
    
    # If we didn't place enough, add random points
    while len(placed) < N:
        rng = np.random.default_rng(len(placed))
        placed.append(rng.uniform(-0.45, 0.45, size=3))
    
    return np.array(placed[:N])


def _polish(x_init, max_iter=2000, time_limit=30.0):
    """
    High-precision polish: focus on active constraints with smaller steps.
    """
    x = x_init.copy().reshape(N, 3)
    x = np.clip(x, -0.49, 0.49)
    
    best_x = x.copy()
    best_score = _min_radius(x)
    
    t0 = time.time()
    
    for it in range(max_iter):
        if time.time() - t0 > time_limit:
            break
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        wall_dists = np.min(np.abs(0.5 - np.abs(x)), axis=1)
        
        min_pair_dist = np.min(dists)
        min_wall_dist = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Find all near-active constraints (within 10% of minimum)
        threshold = current_r * 1.1
        
        # Move all near-active pairs apart slightly
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 2.0 * threshold:
                    direction = x[i] - x[j]
                    norm = np.linalg.norm(direction)
                    if norm < 1e-15:
                        continue
                    direction /= norm
                    step = 0.002
                    x_new = x.copy()
                    x_new[i] += step * direction
                    x_new[j] -= step * direction
                    x_new = np.clip(x_new, -0.49, 0.49)
                    new_score = _min_radius(x_new)
                    if new_score > current_r + 1e-14:
                        x = x_new
                        moved = True
        
        # Move boundary-violating spheres inward
        for i in range(N):
            if wall_dists[i] < threshold:
                direction = -x[i]
                norm = np.linalg.norm(direction)
                if norm < 1e-15:
                    continue
                direction /= norm
                step = 0.002
                x_new = x.copy()
                x_new[i] += step * direction
                x_new = np.clip(x_new, -0.49, 0.49)
                new_score = _min_radius(x_new)
                if new_score > current_r + 1e-14:
                    x = x_new
                    moved = True
        
        if not moved:
            # Try smaller steps
            for i in range(N):
                for j in range(i + 1, N):
                    d = dists[i, j]
                    if d < 2.0 * threshold:
                        direction = x[i] - x[j]
                        norm = np.linalg.norm(direction)
                        if norm < 1e-15:
                            continue
                        direction /= norm
                        step = 0.0005
                        x_new = x.copy()
                        x_new[i] += step * direction
                        x_new[j] -= step * direction
                        x_new = np.clip(x_new, -0.49, 0.49)
                        new_score = _min_radius(x_new)
                        if new_score > current_r + 1e-15:
                            x = x_new
                            moved = True
            
            for i in range(N):
                if wall_dists[i] < threshold:
                    direction = -x[i]
                    norm = np.linalg.norm(direction)
                    if norm < 1e-15:
                        continue
                    direction /= norm
                    step = 0.0005
                    x_new = x.copy()
                    x_new[i] += step * direction
                    x_new = np.clip(x_new, -0.49, 0.49)
                    new_score = _min_radius(x_new)
                    if new_score > current_r + 1e-15:
                        x = x_new
                        moved = True
            
            if not moved:
                break
        
        current_score = _min_radius(x)
        if current_score > best_score:
            best_score = current_score
            best_x = x.copy()
    
    return best_x, best_score


def _scipy_refine(x_init, time_limit=20.0):
    """Use scipy minimize with soft-min objective for refinement."""
    t0 = time.time()
    
    def objective(x):
        return _soft_min_radius(x, 500)
    
    result = minimize(
        objective,
        x_init,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (N * 3),
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    if time.time() - t0 > time_limit:
        return x_init, _min_radius(x_init)
    
    refined_x = np.clip(result.x.reshape(N, 3), -0.5, 0.5)
    score = _min_radius(refined_x)
    return refined_x, score


def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill

    best_x = None
    best_score = -np.inf

    # Multiple initialization strategies
    inits = []
    
    # 1. Fibonacci lattice
    inits.append(_fibonacci_lattice_init(N, seed=42))
    
    # 2. Grid-based
    inits.append(_grid_init(N))
    
    # 3. Random
    inits.append(_random_init(N, seed=123))
    
    # 4. Best-first grid
    inits.append(_best_first_grid_init(N))
    
    # 5. Another random with different seed
    inits.append(_random_init(N, seed=777))
    
    # 6. Fibonacci with different seed
    inits.append(_fibonacci_lattice_init(N, seed=999))

    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline - 30:
            break

        x_init = x_init.copy()
        x_init = np.clip(x_init, -0.48, 0.48)

        # Phase 1: Repulsion-based optimization
        x_refined, score = _repulsion_optimize(x_init, max_iter=3000, time_limit=15.0)
        
        if score > best_score:
            best_score = score
            best_x = x_refined.copy()

        # Phase 2: Scipy refinement
        if time.time() < deadline - 20:
            x_scipy, score_scipy = _scipy_refine(x_refined, time_limit=15.0)
            if score_scipy > best_score:
                best_score = score_scipy
                best_x = x_scipy.copy()

        # Phase 3: Polish
        if time.time() < deadline - 10:
            x_polished, score_polished = _polish(best_x, max_iter=1000, time_limit=8.0)
            if score_polished > best_score:
                best_score = score_polished
                best_x = x_polished.copy()

    # Final polish on the best result
    if best_x is not None and time.time() < deadline - 5:
        x_final, score_final = _polish(best_x, max_iter=500, time_limit=5.0)
        if score_final > best_score:
            best_score = score_final
            best_x = x_final.copy()

    if best_x is None:
        # Fallback to grid
        best_x = _grid_init(N)

    # Final validity check
    best_x = np.clip(best_x, -0.5, 0.5)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-10:
                best_x[j] += 1e-5 * np.array([1, 0, 0])

    return best_x
# EVOLVE-BLOCK-END
