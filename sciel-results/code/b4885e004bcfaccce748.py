"""Heilbronn 11 triangles: Barycentric soft-min optimization with DE exploration."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

DEADLINE = 145  # Safety margin from 150s limit

# Precompute all triplets for N=11 points
# There are C(11, 3) = 165 triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Barycentric coordinates: vertices of the triangle
# V0 = (0, 0), V1 = (1, 0), V2 = (0.5, sqrt(3)/2)
_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, S3H])


def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian.
    bary shape: (N, 3) with each row summing to 1 and non-negative.
    Returns (N, 2) array.
    """
    # point = b0*V0 + b1*V1 + b2*V2
    # = b1*V1 + b2*V2 (since V0 = 0)
    pts = bary[:, 1:2] * _V1[None, :] + bary[:, 2:3] * _V2[None, :]
    return pts


def _cart_to_bary(pts):
    """Convert Cartesian to barycentric coordinates.
    pts shape: (N, 2). Returns (N, 3) array.
    """
    # Solve: p = b0*V0 + b1*V1 + b2*V2, b0+b1+b2=1
    # p = b1*V1 + b2*V2
    # p_x = b1 + 0.5*b2
    # p_y = 0 + S3H*b2
    # So b2 = p_y / S3H, b1 = p_x - 0.5*b2, b0 = 1 - b1 - b2
    b2 = pts[:, 1] / S3H
    b1 = pts[:, 0] - 0.5 * b2
    b0 = 1.0 - b1 - b2
    bary = np.column_stack([b0, b1, b2])
    return bary


def _project_to_triangle(bary):
    """Project barycentric coordinates into the simplex (non-negative, sum=1)."""
    bary = bary.copy()
    # Clip negative values
    bary = np.maximum(bary, 0.0)
    # Normalize to sum to 1
    sums = bary.sum(axis=1, keepdims=True)
    sums = np.maximum(sums, 1e-15)
    bary = bary / sums
    return bary


def _min_triangle_area(bary):
    """Compute the minimum triangle area for all triplets.
    bary shape: (N, 3). Returns the minimum area (not squared).
    """
    pts = _bary_to_cart(bary)
    # Compute areas for all triplets vectorized
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    # Area = 0.5 * |det(pj-pi, pk-pi)|
    d1x = pj[:, 0] - pi[:, 0]
    d1y = pj[:, 1] - pi[:, 1]
    d2x = pk[:, 0] - pi[:, 0]
    d2y = pk[:, 1] - pi[:, 1]
    dets = d1x * d2y - d1y * d2x
    areas = 0.5 * np.abs(dets)
    return np.min(areas)


def _soft_min_area(bary, beta=50.0):
    """Soft minimum of triangle areas using log-sum-exp.
    This is a smooth approximation of the minimum area.
    """
    pts = _bary_to_cart(bary)
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    d1x = pj[:, 0] - pi[:, 0]
    d1y = pj[:, 1] - pi[:, 1]
    d2x = pk[:, 0] - pi[:, 0]
    d2y = pk[:, 1] - pi[:, 1]
    dets = d1x * d2y - d1y * d2x
    areas = 0.5 * np.abs(dets)
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max
    shifted = -beta * areas
    shifted -= np.max(shifted)
    soft_min = -np.log(np.sum(np.exp(shifted))) / beta
    return soft_min


def _bary_objective(bary_flat, beta=50.0):
    """Objective function: negative soft-min area (to maximize)."""
    bary = bary_flat.reshape(N, 3)
    bary = _project_to_triangle(bary)
    return -_soft_min_area(bary, beta)


def _bary_hard_objective(bary_flat):
    """Objective function: negative hard-min area (to maximize)."""
    bary = bary_flat.reshape(N, 3)
    bary = _project_to_triangle(bary)
    return -_min_triangle_area(bary)


def _penalized_hard_min(bary_flat, penalty=10.0):
    """Penalized objective: negative min area plus penalty for out-of-bounds barycentric coords."""
    bary = bary_flat.reshape(N, 3)
    # Penalty for negative barycentric coordinates
    neg_part = np.maximum(-bary, 0.0)
    penalty_val = penalty * np.sum(neg_part ** 2)
    bary_clipped = _project_to_triangle(bary)
    return -_min_triangle_area(bary_clipped) + penalty_val


def _differential_evolution(objective, bounds, pop_size=40, max_iter=100, 
                            mutation=(0.5, 1.0), crossover_rate=0.8, 
                            tol=1e-8, seed=42, deadline=None):
    """Simple differential evolution implementation."""
    rng = np.random.RandomState(seed)
    n_params = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = lb + rng.random((pop_size, n_params)) * (ub - lb)
    fitness = np.array([objective(ind) for ind in pop])
    
    best_idx = np.argmin(fitness)
    best_fit = fitness[best_idx]
    best_ind = pop[best_idx].copy()
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(pop_size):
            if deadline is not None and time.time() > deadline:
                break
            
            # Select three distinct random individuals
            r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
            while r1 == r2 or r2 == r3 or r1 == r3:
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
            
            # Mutation
            F = rng.uniform(*mutation)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover
            cross_mask = rng.random(n_params) < crossover_rate
            if not np.any(cross_mask):
                cross_mask[rng.randint(n_params)] = True
            trial = np.where(cross_mask, mutant, pop[i])
            
            # Clip to bounds
            trial = np.clip(trial, lb, ub)
            
            # Selection
            trial_fit = objective(trial)
            if trial_fit < fitness[i]:
                pop[i] = trial
                fitness[i] = trial_fit
                if trial_fit < best_fit:
                    best_fit = trial_fit
                    best_ind = trial.copy()
                    best_idx = i
    
    return best_ind, best_fit


def _nelder_mead_refine(objective, x0, deadline=None, maxiter=1000):
    """Nelder-Mead refinement."""
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10})
        return result.x, result.fun
    except:
        return x0, objective(x0)


def construct():
    deadline = time.time() + 140  # Safety margin
    
    best_bary = None
    best_val = -np.inf
    
    # Multiple diverse initializations
    init_strategies = []
    
    # 1. Grid-like initialization in barycentric space
    rng = np.random.RandomState(0)
    
    # Strategy 1: Uniform random in barycentric simplex
    for seed in range(5):
        bary = rng.random((N, 3))
        bary = bary / bary.sum(axis=1, keepdims=True)
        init_strategies.append(bary)
    
    # Strategy 2: Near-vertex initialization
    for seed in range(3):
        bary = np.zeros((N, 3))
        for i in range(N):
            # Place near one of the three vertices with some randomness
            v = seed % 3
            bary[i, v] = 0.8 + 0.2 * rng.random()
            bary[i, (v+1)%3] = (1 - bary[i, v]) * 0.5
            bary[i, (v+2)%3] = (1 - bary[i, v]) * 0.5
            bary[i] /= bary[i].sum()
        init_strategies.append(bary)
    
    # Strategy 3: Grid-based in Cartesian, converted to barycentric
    pts = np.array([
        [0.5, 0.0], [0.25, 0.0], [0.75, 0.0],
        [0.25, S3H/4], [0.5, S3H/4], [0.75, S3H/4],
        [0.375, S3H/2], [0.625, S3H/2],
        [0.5, S3H*3/4],
        [0.4375, S3H/8], [0.5625, S3H/8]
    ])
    bary = _cart_to_bary(pts)
    bary = _project_to_triangle(bary)
    init_strategies.append(bary)
    
    # Strategy 4: Sobol-like quasi-random
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=2, seed=123)
    sample = sampler.random(N)
    # Map to triangle: use rejection or mapping
    # Use the standard mapping: if u+v < 1, point is in triangle
    # We'll use a different approach: map [0,1]^2 to triangle
    # x = u + 0.5*v, y = S3H*v, valid when u+v < 1
    # For Latin hypercube, we need to ensure u+v < 1
    # Let's just use random with rejection
    pts = np.zeros((N, 2))
    for i in range(N):
        for attempt in range(100):
            u, v = rng.random(2)
            if u + v < 1.0:
                pts[i, 0] = u + 0.5 * v
                pts[i, 1] = S3H * v
                break
    bary = _cart_to_bary(pts)
    bary = _project_to_triangle(bary)
    init_strategies.append(bary)
    
    # For each initialization, run DE then refine
    for init_idx, bary_init in enumerate(init_strategies):
        if time.time() > deadline - 10:
            break
        
        x0 = bary_init.flatten()
        
        # Phase 1: Differential Evolution with soft-min objective
        beta = 30.0
        obj_soft = lambda x: _bary_objective(x, beta=beta)
        
        # Bounds for barycentric: each coordinate in [0, 1]
        bounds = [(0.0, 1.0)] * (N * 3)
        
        try:
            de_result, de_fit = _differential_evolution(
                obj_soft, bounds, pop_size=30, max_iter=80,
                seed=init_idx, deadline=deadline - 5
            )
            bary_de = _project_to_triangle(de_result.reshape(N, 3))
            val_de = _min_triangle_area(bary_de)
            
            if val_de > best_val:
                best_val = val_de
                best_bary = bary_de.copy()
                
            # Phase 2: Nelder-Mead refinement with hard-min objective
            if time.time() < deadline - 5:
                obj_hard = lambda x: _bary_hard_objective(x)
                nm_x, nm_fit = _nelder_mead_refine(
                    obj_hard, de_result, deadline=deadline - 2, maxiter=500
                )
                bary_nm = _project_to_triangle(nm_x.reshape(N, 3))
                val_nm = _min_triangle_area(bary_nm)
                
                if val_nm > best_val:
                    best_val = val_nm
                    best_bary = bary_nm.copy()
        except Exception as e:
            pass
    
    # Final refinement with multiple Nelder-Mead restarts from best
    if best_bary is not None and time.time() < deadline - 3:
        for restart in range(5):
            if time.time() > deadline - 2:
                break
            # Add small perturbation
            perturbation = np.random.RandomState(restart * 100).normal(0, 0.01, best_bary.shape)
            bary_pert = _project_to_triangle(best_bary + perturbation)
            x_pert = bary_pert.flatten()
            
            obj_hard = lambda x: _bary_hard_objective(x)
            try:
                nm_x, nm_fit = _nelder_mead_refine(
                    obj_hard, x_pert, deadline=deadline - 1, maxiter=300
                )
                bary_final = _project_to_triangle(nm_x.reshape(N, 3))
                val_final = _min_triangle_area(bary_final)
                
                if val_final > best_val:
                    best_val = val_final
                    best_bary = bary_final.copy()
            except:
                pass
    
    if best_bary is None:
        # Fallback: return a valid configuration
        bary = np.random.RandomState(0).random((N, 3))
        bary = bary / bary.sum(axis=1, keepdims=True)
        best_bary = bary
    
    # Convert to Cartesian and return
    pts = _bary_to_cart(best_bary)
    
    # Final validation: ensure all points are inside the triangle
    pts = np.clip(pts, 0, 1)
    # Check and fix any points outside
    for i in range(N):
        x, y = pts[i]
        if y < 0:
            y = 0
        if y > 2 * S3H * x + 1e-9:
            y = 2 * S3H * x
        if y > 2 * S3H * (1 - x) + 1e-9:
            y = 2 * S3H * (1 - x)
        pts[i] = [x, y]
    
    return pts
