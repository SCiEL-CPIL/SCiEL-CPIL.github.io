"""Improved packing for 50 equal circles in a unit square."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(c):
    """Compute the largest common radius for centres c (N,2)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Wall constraints: 1/2 - |c_ik| for each coordinate
    wall = np.min(np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1])))
    if wall < 0:
        return 0.0
    # Pairwise distances
    # Compute all pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist) / 2.0
    return min(wall, min_dist)


def _objective(c, beta):
    """Objective to minimize: -score approximation with soft penalty."""
    c = np.asarray(c, dtype=np.float64)
    wall = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
    min_wall = np.min(wall)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq) / 2.0
    
    # We want to maximize min(min_wall, min_dist)
    # Use a smooth approximation: -max(0, (r_target - min_wall)^2 + (r_target - min_dist)^2)
    # But simpler: minimize -(min_wall + min_dist) / 2 with penalty for imbalance
    # Actually, let's use: minimize -min(min_wall, min_dist) + penalty for constraint violation
    
    r = min(min_wall, min_dist)
    # Penalty for any circle going outside
    pen = 0.0
    for i in range(N):
        for k in range(2):
            v = c[i, k]
            if v > 0.5:
                pen += (v - 0.5) ** 2
            elif v < -0.5:
                pen += (v + 0.5) ** 2
    
    # Penalty for overlapping
    pen2 = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((c[i] - c[j]) ** 2))
            if d < 2 * r:
                pen2 += (2 * r - d) ** 2
    
    return -r + beta * (pen + pen2)


def _construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    best_score = 0.0
    best_c = None
    
    # Generate diverse initial configurations
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # 1. Grid-based (seed)
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initializations.append(grid[:N].copy())
    
    # 2. Hexagonal packing approximation
    hex_c = []
    row = 0
    while len(hex_c) < N:
        x_offset = (row % 2) * 0.5
        for col in range(m + 1):
            x = (col + x_offset) / (m + 0.5) - 0.5
            y = (row + 0.5) / (m + 0.5) - 0.5
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                hex_c.append([x, y])
        row += 1
    if len(hex_c) >= N:
        initializations.append(np.array(hex_c[:N]))
    
    # 3. Random jittered grid
    for trial in range(5):
        c = grid[:N].copy()
        c += rng.normal(0, 0.02, c.shape)
        # Clip to stay in bounds
        c = np.clip(c, -0.48, 0.48)
        initializations.append(c)
    
    # 4. Pure random
    for trial in range(5):
        c = rng.uniform(-0.45, 0.45, (N, 2))
        initializations.append(c)
    
    # 5. Clustering: place circles in groups
    for trial in range(3):
        c = np.zeros((N, 2))
        n_clusters = 5
        centers = rng.uniform(-0.35, 0.35, (n_clusters, 2))
        for i in range(N):
            k = i % n_clusters
            c[i] = centers[k] + rng.normal(0, 0.05, 2)
        c = np.clip(c, -0.48, 0.48)
        initializations.append(c)
    
    # Optimize each initialization
    for init_idx, c0 in enumerate(initializations):
        if time.time() > deadline:
            break
        
        c0 = np.clip(c0, -0.49, 0.49)
        
        # Phase 1: Coarse optimization with moderate beta
        best_local = c0.copy()
        best_local_score = _score(c0)
        
        for beta in [100, 500, 2000, 8000]:
            if time.time() > deadline:
                break
            
            def obj(x):
                return _objective(x, beta)
            
            try:
                res = minimize(
                    obj, c0, method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 1500, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.x is not None and np.all(np.isfinite(res.x)):
                    s = _score(res.x)
                    if s > best_local_score:
                        best_local_score = s
                        best_local = res.x.copy()
                    c0 = res.x.copy()
            except Exception:
                pass
        
        if best_local_score > best_score:
            best_score = best_local_score
            best_c = best_local.copy()
        
        # Phase 2: Polish with high beta
        if time.time() > deadline:
            break
        
        for beta in [10000, 20000]:
            if time.time() > deadline:
                break
            
            def obj(x):
                return _objective(x, beta)
            
            try:
                res = minimize(
                    obj, best_local, method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 3000, 'maxfun': 20000, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                if res.x is not None and np.all(np.isfinite(res.x)):
                    s = _score(res.x)
                    if s > best_score:
                        best_score = s
                        best_c = res.x.copy()
            except Exception:
                pass
    
    if best_c is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    # Verify and return
    final_score = _score(best_c)
    if final_score <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_c


def construct():
    return _construct()
# EVOLVE-BLOCK-END
