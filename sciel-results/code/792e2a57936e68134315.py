"""
Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle.
Strategy:
1. Structured Initialization (Hexagonal, Grid, Symmetric)
2. Differential Evolution for global exploration (robust to non-smooth landscape)
3. L-BFGS-B refinement with log-sum-exp smoothing
4. Nelder-Mead final polish on hard objective
"""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

_SQRT3 = np.sqrt(3.0)
_H = _SQRT3 / 2.0
_EPS = 1e-9

# Precompute all C(11,3) = 165 index triples
_IDX = np.array(list(combinations(range(N), 3)))
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]

def _clamp_bary(bary, eps=1e-8):
    """Clamp barycentric coordinates to valid region: u>=eps, v>=eps, u+v<=1-eps."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
    
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    return np.column_stack([u, v])

def _bary_to_cart(bary):
    """Convert barycentric (u,v) to Cartesian (x,y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * _H
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """Convert Cartesian (x,y) to barycentric (u,v) where w=1-u-v."""
    x = pts[:, 0]
    y = pts[:, 1]
    w = y / _H
    u = 1.0 - w - x
    v = x - 0.5 * w
    return np.column_stack([u, v])

def _areas_bary(bary):
    """Calculate areas of all triangles formed by N points given in barycentric coords (u, v)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[_I]
    vi = v[_I]
    wi = w[_I]
    
    uj = u[_J]
    vj = v[_J]
    wj = w[_J]
    
    uk = u[_K]
    vk = v[_K]
    wk = w[_K]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def _min_area_bary(bary):
    """Minimum triangle area for barycentric points."""
    return float(np.min(_areas_bary(bary)))

def _obj_soft(bary_flat, beta=50.0):
    """Objective function using soft min (log-sum-exp). Returns negative soft-min area."""
    bary = _clamp_bary(bary_flat.reshape(N, 2))
    areas = _areas_bary(bary)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def _obj_hard(bary_flat):
    """Exact non-smooth objective: negative min area."""
    bary = _clamp_bary(bary_flat.reshape(N, 2))
    return -_min_area_bary(bary)

def _generate_seeds():
    """Generate diverse initial configurations in barycentric coordinates."""
    seeds = []
    
    # Seed 1: Hexagonal-like packing
    # Points arranged in rows
    s1 = np.array([
        [0.05, 0.05],
        [0.35, 0.05],
        [0.65, 0.05],
        [0.95, 0.05],
        [0.2, 0.3],
        [0.5, 0.3],
        [0.8, 0.3],
        [0.1, 0.55],
        [0.4, 0.55],
        [0.7, 0.55],
        [0.3, 0.8]
    ])
    seeds.append(_clamp_bary(s1))
    
    # Seed 2: Triangular Grid
    s2 = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.25, 0.25],
        [0.75, 0.25],
        [0.5, 0.5],
        [0.25, 0.5],
        [0.75, 0.5],
        [0.5, 0.75],
        [0.25, 0.75],
        [0.75, 0.75]
    ])
    seeds.append(_clamp_bary(s2))
    
    # Seed 3: Symmetric Center + Perimeter
    s3 = np.array([
        [0.5, 0.5],
        [0.1, 0.1],
        [0.9, 0.1],
        [0.1, 0.8],
        [0.9, 0.8],
        [0.5, 0.1],
        [0.5, 0.9],
        [0.1, 0.5],
        [0.9, 0.5],
        [0.3, 0.3],
        [0.7, 0.7]
    ])
    seeds.append(_clamp_bary(s3))
    
    # Seed 4: Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        seeds.append(_clamp_bary(np.column_stack([u, v])))
        
    return seeds

def _refine_candidate(bary, time_limit=15.0):
    """Refine a candidate using L-BFGS-B then Nelder-Mead."""
    start_time = time.time()
    best_bary = _clamp_bary(np.array(bary, dtype=float).copy())
    best_score = _min_area_bary(best_bary)
    
    bounds = [(0, 1)] * (N * 2)
    
    # Stage 1: L-BFGS-B with soft min
    try:
        if time.time() - start_time < time_limit * 0.6:
            res1 = minimize(
                _obj_soft,
                best_bary.flatten(),
                args=(50.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            cand1 = _clamp_bary(res1.x.reshape(N, 2))
            s1 = _min_area_bary(cand1)
            if s1 > best_score:
                best_score = s1
                best_bary = cand1
    except Exception:
        pass
    
    # Stage 2: Nelder-Mead on hard min
    try:
        if time.time() - start_time < time_limit:
            res2 = minimize(
                _obj_hard,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            cand2 = _clamp_bary(res2.x.reshape(N, 2))
            s2 = _min_area_bary(cand2)
            if s2 > best_score:
                best_score = s2
                best_bary = cand2
    except Exception:
        pass
    
    return best_bary, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_bary = None
    best_score = -1.0
    
    # Generate seeds
    seeds = _generate_seeds()
    
    # Evaluate seeds
    for seed in seeds:
        s = _min_area_bary(seed)
        if s > best_score:
            best_score = s
            best_bary = seed.copy()
            
    # Phase 1: Differential Evolution
    # DE works on flat array. Bounds are (0,1) for all 22 variables.
    bounds = [(0.0, 1.0)] * (N * 2)
    
    # We will run DE for a significant chunk of time.
    # DE can be slow, so we limit maxiter and use a callback to check time.
    
    de_time_limit = 60.0
    de_start = time.time()
    
    def de_callback(xk, convergence):
        if time.time() - de_start > de_time_limit:
            return True
        return False
        
    try:
        # Use a soft objective for DE to help it navigate
        # But we want the final result to be evaluated on hard objective.
        # DE minimizes the function, so we pass _obj_soft.
        res_de = differential_evolution(
            _obj_soft,
            bounds=bounds,
            args=(50.0,),
            maxiter=200,
            popsize=15,
            tol=1e-8,
            seed=42,
            callback=de_callback,
            polish=False  # We will do our own polish
        )
        
        if res_de.x is not None:
            de_bary = _clamp_bary(res_de.x.reshape(N, 2))
            de_score = _min_area_bary(de_bary)
            if de_score > best_score:
                best_score = de_score
                best_bary = de_bary
    except Exception:
        pass
    
    # Phase 2: Refine best candidate and top seeds
    candidates_to_refine = []
    if best_bary is not None:
        candidates_to_refine.append(best_bary)
    candidates_to_refine.extend(seeds[:3]) # Add top seeds
    
    # Deduplicate candidates roughly
    unique_cands = []
    for c in candidates_to_refine:
        is_dup = False
        for uc in unique_cands:
            if np.linalg.norm(c - uc) < 0.1:
                is_dup = True
                break
        if not is_dup:
            unique_cands.append(c)
            
    for cand in unique_cands:
        if time.time() > deadline - 20:
            break
        try:
            ref_bary, ref_score = _refine_candidate(cand, time_limit=15.0)
            if ref_score > best_score:
                best_score = ref_score
                best_bary = ref_bary
        except Exception:
            continue
            
    # Phase 3: Final Polish on the absolute best
    if best_bary is not None and time.time() < deadline - 5:
        try:
            final_bary, final_score = _refine_candidate(best_bary, time_limit=10.0)
            if final_score > best_score:
                best_score = final_score
                best_bary = final_bary
        except Exception:
            pass
            
    # Fallback
    if best_bary is None:
        best_bary = seeds[0]
        
    # Convert to Cartesian and final safety clamp
    result_cart = _bary_to_cart(_clamp_bary(best_bary))
    
    # Final check: ensure points are inside triangle
    # x >= y/sqrt(3) and x <= 1 - y/sqrt(3) and y >= 0
    x = result_cart[:, 0]
    y = result_cart[:, 1]
    y = np.clip(y, 0, _H)
    x_min = y / _SQRT3
    x_max = 1.0 - y / _SQRT3
    x = np.clip(x, x_min, x_max)
    result_cart[:, 0] = x
    result_cart[:, 1] = y
    
    return result_cart

# EVOLVE-BLOCK-END
