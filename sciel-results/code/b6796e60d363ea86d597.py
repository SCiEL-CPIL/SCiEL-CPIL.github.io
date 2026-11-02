"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _get_delaunay_edges(pts):
    """Get Delaunay triangulation edges for proximity-based constraint reduction."""
    try:
        delaunay = Delaunay(pts)
        edges = set()
        for simplex in delaunay.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        # Fallback to all pairs
        n = len(pts)
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _penalty_objective(params, penalty_weight=1000.0):
    """
    Vectorized penalty objective with analytic-friendly structure.
    params: array of length 3*n: [x_0, y_0, r_0, ...]
    Uses soft-max style penalty (squared) for smoothness.
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Wall violations
    wx_left = np.maximum(0.0, r - x)
    wx_right = np.maximum(0.0, r - (1.0 - x))
    wy_bottom = np.maximum(0.0, r - y)
    wy_top = np.maximum(0.0, r - (1.0 - y))
    wall_penalty = np.sum(wx_left**2 + wx_right**2 + wy_bottom**2 + wy_top**2)
    
    # Pairwise overlap violations using Delaunay edges
    pts = np.column_stack([x, y])
    edges = _get_delaunay_edges(pts)
    
    overlap_penalty = 0.0
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i] + r[j] - dist
        if overlap > 0:
            overlap_penalty += overlap ** 2
    
    # Negative radius penalty
    neg_r = np.maximum(0.0, -r)
    neg_penalty = np.sum(neg_r**2)
    
    return -np.sum(r) + penalty_weight * (wall_penalty + overlap_penalty + neg_penalty)

def _penalty_objective_full(params, penalty_weight=1000.0):
    """
    Full pairwise penalty objective (no Delaunay approximation) for final polish.
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Wall violations
    wx_left = np.maximum(0.0, r - x)
    wx_right = np.maximum(0.0, r - (1.0 - x))
    wy_bottom = np.maximum(0.0, r - y)
    wy_top = np.maximum(0.0, r - (1.0 - y))
    wall_penalty = np.sum(wx_left**2 + wx_right**2 + wy_bottom**2 + wy_top**2)
    
    # All pairwise overlaps
    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = r[iu[0]] + r[iu[1]] - dist
    active = overlap > 0
    overlap_penalty = np.sum(overlap[active] ** 2)
    
    # Negative radius penalty
    neg_r = np.maximum(0.0, -r)
    neg_penalty = np.sum(neg_r**2)
    
    return -np.sum(r) + penalty_weight * (wall_penalty + overlap_penalty + neg_penalty)

def _enforce_validity(params):
    """
    Enforce all constraints by shrinking radii and clipping positions.
    Returns a valid configuration.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    # Iteratively resolve overlaps
    for _ in range(200):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        max_ov = 0.0
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            max_ov = max(max_ov, ov)
            # Shrink the smaller circle preferentially
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        # Re-clip to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    return p

def _initial_config_hex():
    """
    Generate an initial configuration using a hexagonal-like arrangement.
    """
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    positions.append([0.5, 0.5, 0.5 / side * 0.5])
    return np.array(positions[:N])

def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_config_voronoi(seed):
    """Initialize using random points with estimated radii."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    
    radii = np.zeros(N)
    for i, p in enumerate(points):
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([p[0], 1 - p[0], p[1], 1 - p[1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.9
        
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params.reshape(N, 3)

def _local_refine(xyz, time_budget=5.0):
    """
    Refine a configuration using L-BFGS-B with Delaunay-based constraints.
    Uses increasing penalty weights for progressive constraint satisfaction.
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_obj = _penalty_objective(best_params, 10.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((1e-6, 0.5))  # r
    
    for penalty_weight in [10.0, 50.0, 100.0, 500.0, 1000.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            return _penalty_objective(p, penalty_weight)
        
        try:
            result = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            obj_val = obj(result.x)
            if obj_val < best_obj:
                best_params = result.x
                best_obj = obj_val
        except Exception:
            pass
    
    return best_params.reshape(N, 3)

def _polish_full(xyz, time_budget=5.0):
    """
    Final high-precision polish using full pairwise constraints (no Delaunay approximation).
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_obj = _penalty_objective_full(best_params, 1000.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    for penalty_weight in [500.0, 1000.0, 5000.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            return _penalty_objective_full(p, penalty_weight)
        
        try:
            result = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            obj_val = obj(result.x)
            if obj_val < best_obj:
                best_params = result.x
                best_obj = obj_val
        except Exception:
            pass
    
    return best_params.reshape(N, 3)

def _cma_es_search(xyz0, time_budget=15.0, sigma0=0.05):
    """
    CMA-ES style evolutionary search to escape local minima.
    Simple implementation using population-based search with penalty objective.
    """
    start_time = time.time()
    x = xyz0.flatten().copy()
    sigma = sigma0
    
    # Population size
    pop_size = max(10, N // 2)
    
    best_x = x.copy()
    best_obj = _penalty_objective(best_x, 100.0)
    
    # Bounds
    lo = np.array([0.0 if i % 3 != 2 else 1e-6 for i in range(3 * N)])
    hi = np.array([1.0 if i % 3 != 2 else 0.5 for i in range(3 * N)])
    
    generations = 0
    while time.time() - start_time < time_budget:
        generations += 1
        
        # Generate population
        population = []
        for _ in range(pop_size):
            mutant = x + sigma * np.random.randn(3 * N)
            mutant = np.clip(mutant, lo, hi)
            population.append(mutant)
        
        # Evaluate
        objs = [_penalty_objective(p, 100.0) for p in population]
        
        # Select best
        best_idx = np.argmin(objs)
        if objs[best_idx] < best_obj:
            best_obj = objs[best_idx]
            best_x = population[best_idx].copy()
        
        # Update mean (weighted by rank)
        sorted_indices = np.argsort(objs)
        weights = np.exp(-np.arange(pop_size) / (pop_size / 2))
        weights /= np.sum(weights)
        new_mean = np.zeros(3 * N)
        for w, idx in zip(weights, sorted_indices):
            new_mean += w * population[idx]
        x = np.clip(new_mean, lo, hi)
        
        # Adapt sigma (simple version)
        sigma *= 0.95
        sigma = max(sigma, 0.001)
        
        if time.time() - start_time > time_budget * 0.8:
            break
    
    return best_x.reshape(N, 3)

def _optimize_hybrid(xyz, time_budget=20.0):
    """
    Hybrid optimization: CMA-ES for global exploration, L-BFGS-B for local refinement,
    and full pairwise polish for final precision.
    """
    start_time = time.time()
    params = xyz.flatten()
    
    # Phase 1: CMA-ES for global exploration
    cma_time = min(8.0, time_budget * 0.4)
    if time_budget > 10.0:
        try:
            cma_result = _cma_es_search(params.reshape(N, 3), time_budget=cma_time, sigma0=0.05)
            cma_obj = _penalty_objective(cma_result.flatten(), 100.0)
            orig_obj = _penalty_objective(params, 100.0)
            if cma_obj < orig_obj:
                params = cma_result.flatten()
        except Exception:
            pass
    
    # Phase 2: Local Refinement with L-BFGS-B (Delaunay-based)
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 2.0:
        xyz_refined = _local_refine(params.reshape(N, 3), time_budget=remaining_time * 0.6)
        if np.sum(xyz_refined[:, 2]) > np.sum(params[2::3]):
            params = xyz_refined.flatten()
    
    # Phase 3: Full pairwise polish
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 1.0:
        xyz_polished = _polish_full(params.reshape(N, 3), time_budget=remaining_time * 0.8)
        if np.sum(xyz_polished[:, 2]) > np.sum(params[2::3]):
            params = xyz_polished.flatten()
    
    return params.reshape(N, 3)

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global_start = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Try multiple initial configurations
    initial_configs = []
    
    # 1. Simple grid
    initial_configs.append(_initial_config_grid())
    
    # 2. Hexagonal-like
    initial_configs.append(_initial_config_hex())
    
    # 3. Voronoi-based random
    for seed in [42, 123, 456]:
        initial_configs.append(_initial_config_voronoi(seed))
    
    # 4. Random configurations
    np.random.seed(42)
    for trial in range(3):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    # 5. Concentric rings pattern
    positions = []
    positions.append([0.5, 0.5, 0.10])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.12 * 1.8 * np.cos(angle)
        y = 0.5 + 0.12 * 1.8 * np.sin(angle)
        positions.append([x, y, 0.12 * 0.9])
    for k in range(10):
        angle = 2 * np.pi * k / 10 + np.pi / 10
        x = 0.5 + 0.10 * 3.5 * np.cos(angle)
        y = 0.5 + 0.10 * 3.5 * np.sin(angle)
        positions.append([x, y, 0.10 * 0.9])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.08 * 4.8 * np.cos(angle)
        y = 0.5 + 0.08 * 4.8 * np.sin(angle)
        positions.append([x, y, 0.08 * 0.9])
    
    initial_configs.append(np.array(positions[:N]))
    
    # Optimize from each initial configuration using hybrid method
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        time_budget = min(20.0, remaining - 2.0)
        if time_budget < 2.0:
            break
        
        try:
            xyz_opt = _optimize_hybrid(xyz0, time_budget=time_budget)
            # Enforce validity
            xyz_fixed = _enforce_validity(xyz_opt.flatten())
            
            if _validate(xyz_fixed):
                score = np.sum(xyz_fixed[:, 2])
                if score > best_score:
                    best_score = score
                    best_xyz = xyz_fixed
            else:
                # Fallback: shrink further
                xyz_fixed2 = xyz_fixed.copy()
                for _ in range(100):
                    if _validate(xyz_fixed2):
                        break
                    xyz_fixed2[:, 2] *= 0.9
                    for i in range(N):
                        x, y, r = xyz_fixed2[i]
                        xyz_fixed2[i, 0] = np.clip(x, r, 1 - r)
                        xyz_fixed2[i, 1] = np.clip(y, r, 1 - r)
                if _validate(xyz_fixed2):
                    score = np.sum(xyz_fixed2[:, 2])
                    if score > best_score:
                        best_score = score
                        best_xyz = xyz_fixed2
        except Exception:
            pass
    
    # Final check and polish
    if not _validate(best_xyz):
        best_xyz = _enforce_validity(best_xyz.flatten())
    
    # Final high-precision polish if time allows
    elapsed = time.time() - global_start
    remaining = deadline - elapsed
    if remaining > 3.0 and _validate(best_xyz):
        try:
            polished = _polish_full(best_xyz, time_budget=remaining * 0.8)
            polished_valid = _enforce_validity(polished.flatten())
            if _validate(polished_valid) and np.sum(polished_valid[:, 2]) > np.sum(best_xyz[:, 2]):
                best_xyz = polished_valid
        except Exception:
            pass
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _enforce_validity(best_xyz.flatten())
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
