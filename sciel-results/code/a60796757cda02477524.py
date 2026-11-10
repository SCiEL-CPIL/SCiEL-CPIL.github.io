"""Improved packing of 50 equal spheres in a unit cube using KDTree + L-BFGS-B + local search."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _exact_score(centers):
    """Exact minimum radius score using KDTree for speed."""
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    return min(min_pair / 2.0, min_wall)

def _soft_min_score(centers, beta=50.0):
    """Differentiable soft-minimum score for gradient-based optimization."""
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    a = min_pair / 2.0
    b = min_wall
    # Soft min: -1/beta * log(exp(-beta*a) + exp(-beta*b))
    # Use log-sum-exp trick for stability
    m = min(a, b)
    return m - math.log(math.exp(-beta * (a - m)) + math.exp(-beta * (b - m))) / beta

def _objective_flat(x, beta=50.0):
    """Objective for L-BFGS-B: minimize negative soft-min score."""
    centers = x.reshape(N, 3)
    centers = np.clip(centers, -0.5, 0.5)
    return -_soft_min_score(centers, beta=beta)

def _initial_grid_pfoertner():
    """5x5x2 grid inspired by Pfoertner's packing."""
    centers = np.zeros((N, 3))
    m = 5
    spacing = 1.0 / (m + 1)
    idx = 0
    for layer in range(2):
        z = -0.5 + (layer + 0.5) * 0.5
        for i in range(m):
            for j in range(m):
                x = -0.5 + (i + 0.5) * spacing
                y = -0.5 + (j + 0.5) * spacing
                centers[idx] = [x, y, z]
                idx += 1
    return centers

def _initial_cubic_grid():
    """Cubic grid initializer."""
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _fast_local_search(centers, max_iter=50000, step_size=0.02, seed=42, time_limit=10.0):
    """Fast local search using KDTree scoring."""
    rng = np.random.RandomState(seed)
    current = centers.copy()
    best = current.copy()
    best_score = _exact_score(best)
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        idx = rng.randint(N)
        delta = rng.uniform(-step_size, step_size, size=3)
        candidate = current.copy()
        candidate[idx] += delta
        
        if np.any(np.abs(candidate[idx]) > 0.5):
            continue
        
        new_score = _exact_score(candidate)
        if new_score > best_score:
            best = candidate.copy()
            best_score = new_score
            current = candidate
    
    return best, best_score

def _l_bfgs_refine(centers, time_limit=15.0):
    """Refine using L-BFGS-B with soft-minimum objective."""
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Coarse pass
    if time.time() - start_time < time_limit * 0.4:
        result1 = minimize(
            _objective_flat, x0, args=(20.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        x0 = result1.x
    
    # Fine pass
    if time.time() - start_time < time_limit * 0.8:
        result2 = minimize(
            _objective_flat, x0, args=(80.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        x0 = result2.x
    
    return np.clip(x0.reshape(N, 3), -0.5, 0.5)

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # Multiple initializers
    initializers = []
    pfoertner = _initial_grid_pfoertner()
    initializers.append(("pfoertner", pfoertner))
    
    cubic = _initial_cubic_grid()
    initializers.append(("cubic", cubic))
    
    # Perturbed versions
    for seed in range(4):
        rng = np.random.RandomState(seed * 100)
        pert = pfoertner + rng.uniform(-0.015, 0.015, size=(N, 3))
        pert = np.clip(pert, -0.5, 0.5)
        initializers.append((f"pfoertner_pert_{seed}", pert))
    
    best_centers = pfoertner.copy()
    best_score = _exact_score(best_centers)
    
    # Phase 1: Fast local search on each initializer
    for name, init_c in initializers:
        if time.time() > deadline - 30.0:
            break
        remaining = deadline - 30.0 - time.time()
        ls_time = min(8.0, remaining * 0.5)
        if ls_time < 1.0:
            break
        
        ls_c, ls_s = _fast_local_search(init_c, max_iter=50000, step_size=0.02, 
                                         seed=hash(name) % 10000, time_limit=ls_time)
        if ls_s > best_score:
            best_score = ls_s
            best_centers = ls_c.copy()
    
    # Phase 2: L-BFGS-B refinement on the best solution
    if time.time() < deadline - 20.0:
        remaining = deadline - 20.0 - time.time()
        refine_time = min(15.0, remaining * 0.6)
        if refine_time > 2.0:
            refined = _l_bfgs_refine(best_centers, time_limit=refine_time)
            refined_score = _exact_score(refined)
            if refined_score > best_score:
                best_score = refined_score
                best_centers = refined.copy()
    
    # Phase 3: Another round of local search from refined solution
    if time.time() < deadline - 10.0:
        remaining = deadline - 10.0 - time.time()
        ls_time = min(6.0, remaining * 0.7)
        if ls_time > 1.0:
            ls_c, ls_s = _fast_local_search(best_centers, max_iter=50000, step_size=0.01,
                                             seed=99999, time_limit=ls_time)
            if ls_s > best_score:
                best_score = ls_s
                best_centers = ls_c.copy()
    
    # Phase 4: Final L-BFGS-B polish
    if time.time() < deadline - 5.0:
        remaining = deadline - 5.0 - time.time()
        refine_time = min(8.0, remaining * 0.7)
        if refine_time > 1.0:
            refined = _l_bfgs_refine(best_centers, time_limit=refine_time)
            refined_score = _exact_score(refined)
            if refined_score > best_score:
                best_score = refined_score
                best_centers = refined.copy()
    
    # Final local search polish
    if time.time() < deadline - 2.0:
        remaining = deadline - 2.0 - time.time()
        ls_time = min(3.0, remaining * 0.8)
        if ls_time > 0.5:
            ls_c, ls_s = _fast_local_search(best_centers, max_iter=20000, step_size=0.005,
                                             seed=12345, time_limit=ls_time)
            if ls_s > best_score:
                best_score = ls_s
                best_centers = ls_c.copy()
    
    return best_centers

# EVOLVE-BLOCK-END
