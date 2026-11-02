"""
Optimized packing of 50 equal circles in a unit square.
Combines soft-min (log-sum-exp) L-BFGS polishing with progressive tau reduction
and multi-start diversity to directly maximize the minimum distance metric.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 50
HALF = 0.5
# Precompute upper-triangle indices for pairwise operations
_IU = np.triu_indices(N, 1)
_NPAIRS = len(_IU[0])


def _score(c):
    """Largest common radius for centres c (N, 2)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    diff = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0
    min_bound = np.min(HALF - np.abs(c))
    return min(min_pair, min_bound)


def _soft_min_obj_and_grad(x, tau):
    """
    Objective: minimize -soft_min(all constraint distances).
    Uses log-sum-exp for smooth approximation of min.
    Returns (f, g) where f = -soft_min, g = gradient.
    """
    centers = x.reshape(-1, 2)

    # Wall distances per circle
    wall_dists_x = HALF - np.abs(centers[:, 0])
    wall_dists_y = HALF - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)  # (N,)

    # Pairwise distances (upper triangle)
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))  # (_NPAIRS,)
    pair_dists = dists / 2.0  # radius contribution

    # All constraint distances: N walls + NPAIRS pairs
    all_dists = np.concatenate([walls, pair_dists])  # (N + _NPAIRS,)

    # Soft-min via log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    soft_min = min_d - tau * np.log(sum_exp)

    f = -soft_min

    # Gradient
    grad = np.zeros_like(x)

    # Wall gradient contributions
    for i in range(N):
        w = weights[i]
        if w < 1e-12:
            continue
        if wall_dists_x[i] < wall_dists_y[i]:
            grad[2 * i] += w * np.sign(centers[i, 0])
        elif wall_dists_y[i] < wall_dists_x[i]:
            grad[2 * i + 1] += w * np.sign(centers[i, 1])
        else:
            grad[2 * i] += 0.5 * w * np.sign(centers[i, 0])
            grad[2 * i + 1] += 0.5 * w * np.sign(centers[i, 1])

    # Pair gradient contributions (vectorized)
    pair_weights = weights[N:]  # (_NPAIRS,)
    dists_safe = np.maximum(dists, 1e-12)
    factor = pair_weights / dists_safe  # (_NPAIRS,)
    contrib = factor[:, None] * diff  # (_NPAIRS, 2)

    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])

    return f, grad


def _initial_grid():
    """Create initial grid layout with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()

    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -HALF + 0.01, HALF - 0.01)

    return grid


def _hex_init():
    """Create a hexagonal packing initial layout."""
    best_grid = None
    best_r = 0

    rng = np.random.default_rng(42)

    for attempt in range(8):
        rows = rng.integers(5, 9)
        cols = int(np.ceil(N / rows))

        s_width = 1.0 / (cols + 0.3)
        s_height = 1.0 / (rows * 0.866 + 0.3)
        s = min(s_width, s_height)

        centers = []
        for j in range(rows):
            for i in range(cols):
                x = (i - (cols - 1) / 2) * s
                y = (j - (rows - 1) / 2) * s * 0.866
                if j % 2 == 1:
                    x += s / 2
                centers.append([x, y])

        centers = np.array(centers[:N])
        if len(centers) < N:
            while len(centers) < N:
                centers = np.vstack([centers, rng.uniform(-0.45, 0.45, 2)])

        centers = np.clip(centers, -HALF + 0.01, HALF - 0.01)
        r = _score(centers)
        if r > best_r:
            best_r = r
            best_grid = centers.copy()

    return best_grid if best_grid is not None else _initial_grid()


def _laplacian_smooth(centers, time_budget):
    """
    Minimize Dirichlet energy of the k-NN graph to enforce uniform spacing.
    This acts as a regularizer to prevent clustering and improve the initial
    layout before the main soft-min optimization.
    """
    start_time = time.time()
    
    # 1. Build k-NN graph (k=6 is good for hex)
    tree = cKDTree(centers)
    _, idx = tree.query(centers, k=7)  # idx shape (N, 7), idx[:,0] is self
    edges = set()
    for i in range(N):
        for j in idx[i, 1:]:
            if i != j:
                if i < j:
                    edges.add((i, j))
                else:
                    edges.add((j, i))
    
    if not edges:
        return centers
    
    i_idx = np.array([e[0] for e in edges])
    j_idx = np.array([e[1] for e in edges])
    
    # 2. Determine target length
    ci = centers[i_idx]
    cj = centers[j_idx]
    dists = np.linalg.norm(ci - cj, axis=1)
    target_len = np.median(dists)
    
    # 3. Optimize elastic energy
    def obj_and_grad(x):
        if time.time() > start_time + time_budget:
            return 0.0, np.zeros_like(x)
        
        c = x.reshape(-1, 2)
        ci = c[i_idx]
        cj = c[j_idx]
        diff = ci - cj
        dists = np.linalg.norm(diff, axis=1)
        errors = dists - target_len
        f = np.sum(errors ** 2)
        
        grad = np.zeros_like(x)
        dists_safe = np.maximum(dists, 1e-12)
        factor = 2.0 * errors / dists_safe
        contrib = factor[:, None] * diff
        
        np.add.at(grad, 2 * i_idx, contrib[:, 0])
        np.add.at(grad, 2 * i_idx + 1, contrib[:, 1])
        np.add.at(grad, 2 * j_idx, -contrib[:, 0])
        np.add.at(grad, 2 * j_idx + 1, -contrib[:, 1])
        
        return f, grad
    
    try:
        res = minimize(
            obj_and_grad,
            centers.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 50, 'gtol': 1e-6, 'ftol': 1e-8}
        )
        new_centers = res.x.reshape(N, 2)
        new_centers = np.clip(new_centers, -HALF + 0.01, HALF - 0.01)
        return new_centers
    except Exception:
        return centers


def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill

    # Generate diverse initial seeds
    seeds = [
        _hex_init(),
        _initial_grid()
    ]
    
    # Add random seeds
    rng = np.random.default_rng(0)
    for _ in range(3):
        seed = rng.uniform(-HALF + 0.05, HALF - 0.05, (N, 2))
        seeds.append(seed)

    best_pts = seeds[0].copy()
    best_r = _score(best_pts)

    # Phase 1: Laplacian smoothing to improve initial layouts
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 60:
            break
        smoothed = _laplacian_smooth(seed, time_budget=5.0)
        r = _score(smoothed)
        if r > best_r:
            best_r = r
            best_pts = smoothed.copy()

    # Phase 2: Progressive soft-min optimization with decreasing tau
    # Start with larger tau for robustness, then decrease for precision
    taus = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    current_pts = best_pts.copy()
    
    for tau in taus:
        if time.time() > deadline - 10:
            break
        
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                current_pts.flatten(),
                args=(tau,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            
            new_pts = res.x.reshape(N, 2)
            new_pts = np.clip(new_pts, -HALF, HALF)
            
            new_r = _score(new_pts)
            
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
            
            # Use the result as starting point for next tau
            current_pts = new_pts.copy()
            
        except Exception:
            # If optimization fails, try restarting from best_pts
            current_pts = best_pts.copy()
            continue

    # Phase 3: Final high-precision polish with very small tau
    if time.time() < deadline - 5:
        for tau in [0.0002, 0.0001, 0.00005]:
            if time.time() > deadline - 2:
                break
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    best_pts.flatten(),
                    args=(tau,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'gtol': 1e-12, 'ftol': 1e-14}
                )
                
                new_pts = res.x.reshape(N, 2)
                new_pts = np.clip(new_pts, -HALF, HALF)
                
                new_r = _score(new_pts)
                
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
                    
            except Exception:
                continue

    # Final sanity check
    final_r = _score(best_pts)
    if final_r > 0:
        return best_pts
    else:
        # Fallback to a valid grid
        return _initial_grid()
