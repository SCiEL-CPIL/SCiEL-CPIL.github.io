"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START

_DEADLINE = 130.0  # wall-clock safety margin before 150s kill


def _objective_flat(params, n):
    """Flat parameter vector: positions (2n) then radii (n).
    Returns a cost (negative of sum of radii minus penalties) to minimize."""
    pos = params[:2 * n]
    rad = params[2 * n:]
    
    # Penalty for circles outside [0,1]^2
    penalty = 0.0
    # x bounds
    penalty += np.sum(np.maximum(0, rad - pos[0::2]) ** 2) + np.sum(np.maximum(0, (1 - pos[0::2]) - rad) ** 2)
    # y bounds
    penalty += np.sum(np.maximum(0, rad - pos[1::2]) ** 2) + np.sum(np.maximum(0, (1 - pos[1::2]) - rad) ** 2)
    
    # Penalty for overlapping circles
    # Vectorized pairwise distances
    x = pos[0::2]
    y = pos[1::2]
    # Compute all pairwise squared distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    rsum = rad[:, None] + rad[None, :]
    # Only consider i < j (upper triangle)
    iu = np.triu_indices(n, k=1)
    overlap = np.maximum(0, rsum[iu] - np.sqrt(d2[iu]))
    penalty += np.sum(overlap ** 2)
    
    # Objective: minimize penalty - sum(rad)  (negative sum to maximize)
    return penalty - np.sum(rad)


def _objective_pos_only(params, rad, n):
    """Minimize penalty given fixed radii; params are positions only (2n)."""
    pos = params
    x = pos[0::2]
    y = pos[1::2]
    
    # Wall penalty
    penalty = np.sum(np.maximum(0, rad - x) ** 2) + np.sum(np.maximum(0, (1 - x) - rad) ** 2)
    penalty += np.sum(np.maximum(0, rad - y) ** 2) + np.sum(np.maximum(0, (1 - y) - rad) ** 2)
    
    # Overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    rsum = rad[:, None] + rad[None, :]
    iu = np.triu_indices(n, k=1)
    overlap = np.maximum(0, rsum[iu] - np.sqrt(d2[iu]))
    penalty += np.sum(overlap ** 2)
    
    return penalty


def _make_flat_obj(n):
    def f(params):
        return _objective_flat(params, n)
    return f


def _construct_grid(n):
    """Return flat params (pos + radii) for a grid-like initial guess."""
    side = int(np.ceil(np.sqrt(n)))
    # Place n circles in a side x side grid
    xs = []
    ys = []
    for i in range(n):
        col = i % side
        row = i // side
        xs.append((col + 0.5) / side)
        ys.append((row + 0.5) / side)
    xs = np.array(xs)
    ys = np.array(ys)
    # Initial radius estimate
    r = 0.4 / side * 0.95
    rad = np.full(n, r)
    pos = np.concatenate([xs, ys])
    return np.concatenate([pos, rad])


def construct():
    start = time.time()
    n = N
    
    # Multi-start with different initializations, each run through
    # differential evolution (or L-BFGS-B for speed) then local polish.
    
    best_cost = np.inf
    best_params = None
    
    # Helper to check deadline
    def _time_left():
        return _DEADLINE - (time.time() - start)
    
    # --- Phase 1: Several random/grid starts with L-BFGS-B on full params ---
    rng = np.random.default_rng(42)
    
    n_starts = 8  # number of multi-starts
    for s in range(n_starts):
        if _time_left() < 15:
            break
        
        # Generate initial params
        if s == 0:
            params0 = _construct_grid(n)
        elif s == 1:
            # Slightly perturbed grid
            params0 = _construct_grid(n) + rng.normal(0, 0.05, size=3 * n)
        else:
            # Random positions and radii
            xs = rng.uniform(0.1, 0.9, n)
            ys = rng.uniform(0.1, 0.9, n)
            rad = rng.uniform(0.02, 0.12, n)
            pos = np.concatenate([xs, ys])
            params0 = np.concatenate([pos, rad])
        
        obj = _make_flat_obj(n)
        # Bounds: x,y in [0,1], r in [0, 0.5]
        bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
        
        try:
            res = minimize(obj, params0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success and res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # --- Phase 2: Differential Evolution for global search (if time allows) ---
    if _time_left() > 30:
        obj = _make_flat_obj(n)
        bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
        try:
            res_de = differential_evolution(obj, bounds, seed=42, maxiter=200,
                                             popsize=15, tol=1e-8, 
                                             polish=True, init='sobol')
            if res_de.fun < best_cost:
                best_cost = res_de.fun
                best_params = res_de.x.copy()
        except Exception:
            pass
    
    # --- Phase 3: Local polish with fixed radii optimization ---
    if best_params is not None and _time_left() > 10:
        pos = best_params[:2 * n].copy()
        rad = best_params[2 * n:].copy()
        
        # Iterate: optimize positions, then scale radii up as much as possible
        for it in range(50):
            if _time_left() < 10:
                break
            
            # Optimize positions with current radii
            def pos_obj(p):
                return _objective_pos_only(p, rad, n)
            
            try:
                res_p = minimize(pos_obj, pos, method='L-BFGS-B',
                                 bounds=[(0, 1)] * (2 * n),
                                 options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
                if res_p.fun < _objective_pos_only(pos, rad, n) - 1e-15:
                    pos = res_p.x.copy()
            except Exception:
                break
            
            # Try to increase radii: compute max feasible radius for each circle
            # Given positions, the max radius for circle i is limited by walls and other circles
            x = pos[0::2]
            y = pos[1::2]
            max_rad = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                    # r_i + r_j <= d  =>  r_i <= d - r_j
                    # But we want to increase r_i; the constraint is r_i <= d - r_j
                    # To be safe, use current r_j
                    limit = d - rad[j]
                    if limit < max_rad[i]:
                        max_rad[i] = limit
            
            # Scale up radii toward max_rad
            scale_factor = 0.1  # fraction of the gap to take
            for i in range(n):
                if max_rad[i] > rad[i]:
                    rad[i] += scale_factor * (max_rad[i] - rad[i])
                else:
                    rad[i] = max(0.001, rad[i] * 0.99)  # shrink if over
            rad = np.clip(rad, 0.001, 0.5)
        
        # Final: assemble and re-optimize full params one more time
        final_params = np.concatenate([pos, rad])
        obj = _make_flat_obj(n)
        bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
        try:
            res_final = minimize(obj, final_params, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res_final.fun < best_cost:
                best_cost = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    # Extract final result
    if best_params is None:
        # Fallback: grid
        return np.array(_construct_grid(n)).reshape(n, 3)[:, [0, 1, 2]]
    
    pos = best_params[:2 * n]
    rad = best_params[2 * n:]
    
    # Assemble (x, y, radius)
    result = np.column_stack([pos[0::2], pos[1::2], rad])
    
    # Safety clip: ensure inside [0,1]^2
    result[:, 0] = np.clip(result[:, 0], result[:, 2], 1 - result[:, 2])
    result[:, 1] = np.clip(result[:, 1], result[:, 2], 1 - result[:, 2])
    
    return result

# EVOLVE-BLOCK-END
