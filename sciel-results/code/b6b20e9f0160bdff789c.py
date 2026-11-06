"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _radius(centers):
    """Compute max common radius (vectorized)."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_dist = np.sqrt(np.min(d2))
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair_dist / 2.0, min_wall)


def _bottleneck_pairs(centers, top_k=5):
    """Return the top_k closest pairs and wall-constrained indices."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    # Get indices of smallest distances
    flat_idx = np.argsort(d2.ravel())[:top_k * 2]
    pairs = set()
    for idx in flat_idx:
        i, j = divmod(idx, n)
        if i < j:
            pairs.add((i, j))
        elif j < i:
            pairs.add((j, i))
    pair_list = list(pairs)[:top_k]
    
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    wall_idx = np.argsort(wall)[:top_k]
    return pair_list, list(wall_idx)


def _relieve_pair(centers, i, j, step=0.01):
    """Move spheres i and j apart along their connecting vector."""
    c = centers.copy()
    diff = c[i] - c[j]
    d = np.linalg.norm(diff)
    if d < 1e-12:
        diff = np.random.randn(3) * 1e-3
        d = np.linalg.norm(diff)
    unit = diff / d
    c[i] += step * unit
    c[j] -= step * unit
    return c


def _relieve_wall(centers, idx, step=0.01):
    """Move sphere idx away from the nearest wall."""
    c = centers.copy()
    # Find which wall is closest
    for k in range(3):
        if c[idx, k] > 0:
            # Near +x or +y or +z wall
            if 0.5 - c[idx, k] == min(0.5 - abs(c[idx, m]) for m in range(3)):
                c[idx, k] -= step
        else:
            if 0.5 + c[idx, k] == min(0.5 - abs(c[idx, m]) for m in range(3)):
                c[idx, k] += step
    return c


def _targeted_refine(centers, time_budget, seed=42):
    """Iteratively relieve bottleneck pairs and wall constraints."""
    start = time.time()
    c = centers.copy()
    best_r = _radius(c)
    best = c.copy()
    rng = np.random.RandomState(seed)
    
    step = 0.02
    patience = 0
    
    while time.time() - start < time_budget:
        pairs, wall_idx = _bottleneck_pairs(c, top_k=3)
        
        improved = False
        for i, j in pairs:
            new_c = _relieve_pair(c, i, j, step)
            new_c = np.clip(new_c, -0.499, 0.499)
            r = _radius(new_c)
            if r > best_r + 1e-10:
                c = new_c
                best_r = r
                best = c.copy()
                improved = True
                break
            elif r > _radius(c) + 1e-10:
                c = new_c
                improved = True
                break
        
        if not improved:
            for idx in wall_idx:
                new_c = _relieve_wall(c, idx, step)
                new_c = np.clip(new_c, -0.499, 0.499)
                r = _radius(new_c)
                if r > best_r + 1e-10:
                    c = new_c
                    best_r = r
                    best = c.copy()
                    improved = True
                    break
                elif r > _radius(c) + 1e-10:
                    c = new_c
                    improved = True
                    break
        
        if not improved:
            patience += 1
            step *= 0.9
            if patience > 50 or step < 1e-5:
                break
        else:
            patience = 0
    
    return best, best_r


def _nelder_mead_polish(centers, time_budget):
    """Light Nelder-Mead polish on the full configuration."""
    from scipy.optimize import minimize
    start = time.time()
    
    def objective(x):
        c = x.reshape(N, 3)
        if np.any(np.abs(c) > 0.499) or not np.all(np.isfinite(c)):
            return 1e10
        return -_radius(c)
    
    x0 = centers.flatten()
    try:
        result = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        c = result.x.reshape(N, 3)
        if np.all(np.abs(c) <= 0.499) and np.all(np.isfinite(c)):
            r = _radius(c)
            if r > 0:
                return c, r
    except Exception:
        pass
    return centers, _radius(centers)


def construct():
    """Construct 50 sphere centers in a unit cube for maximum radius."""
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Start from the simple cubic grid (known to score 1.0)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Small deterministic perturbation to break exact symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_r = _radius(best_centers)
    
    # Phase 1: Targeted bottleneck relief (preserves grid structure)
    if time.time() - start_time < deadline - 20:
        budget = min(15.0, deadline - 20 - (time.time() - start_time))
        refined, r = _targeted_refine(best_centers, budget, seed=42)
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
    
    # Phase 2: Second targeted pass with different seed
    if time.time() - start_time < deadline - 10:
        budget = min(10.0, deadline - 10 - (time.time() - start_time))
        refined, r = _targeted_refine(best_centers, budget, seed=123)
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
    
    # Phase 3: Nelder-Mead polish (careful, may help or hurt)
    if time.time() - start_time < deadline - 3:
        budget = min(8.0, deadline - 3 - (time.time() - start_time))
        polished, r = _nelder_mead_polish(best_centers, budget)
        if r > best_r:
            best_r = r
            best_centers = polished.copy()
    
    # Final validity check
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.499):
        best_centers = grid[:N].copy()
    
    return best_centers
# EVOLVE-BLOCK-END
