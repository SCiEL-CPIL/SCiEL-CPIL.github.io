# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product
from scipy.optimize import minimize

def _check_validity(centers, sides, tol=1e-7):
    """Check if the configuration is valid."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                return False
    return True

def _get_compositions(n, k):
    """Generate all compositions of n into k positive integers."""
    if k == 1:
        yield (n,)
        return
    for i in range(1, n - k + 2):
        for rest in _get_compositions(n - i, k - 1):
            yield (i,) + rest

def _compute_grid_objective(params, comps):
    """
    Compute the negative sum of square sides for a given grid configuration.
    params: flattened vector of cuts.
    comps: tuple of square counts per column.
    """
    C = len(comps)
    # Unpack params
    # Structure: [x_cuts (C-1), y_cuts_col0 (n0-1), y_cuts_col1 (n1-1), ...]
    
    idx = 0
    x_cuts = np.zeros(C-1)
    for i in range(C-1):
        x_cuts[i] = params[idx]
        idx += 1
        
    y_cuts_list = []
    for c in range(C):
        n_c = comps[c]
        n_cuts = n_c - 1
        y_cuts = np.zeros(n_cuts)
        for j in range(n_cuts):
            y_cuts[j] = params[idx]
            idx += 1
        y_cuts_list.append(y_cuts)
        
    # Calculate column widths
    x_bounds = [0.0] + list(x_cuts) + [1.0]
    col_widths = np.diff(x_bounds)
    
    total_side_sum = 0.0
    centers = []
    sides = []
    
    for c in range(C):
        w = col_widths[c]
        n_c = comps[c]
        y_cuts = y_cuts_list[c]
        y_bounds = [0.0] + list(y_cuts) + [1.0]
        row_heights = np.diff(y_bounds)
        
        for r in range(n_c):
            h = row_heights[r]
            s = min(w, h)
            total_side_sum += s
            
            # Calculate center
            cx = x_bounds[c] + w / 2.0
            cy = y_bounds[r] + h / 2.0
            centers.append([cx, cy])
            sides.append(s)
            
    # Return negative for minimization
    return -total_side_sum, np.array(centers), np.array(sides)

def _optimize_composition(comps, n_restarts=50, time_limit=2.0):
    """
    Optimize the cuts for a specific composition.
    Returns best (score, centers, sides).
    """
    C = len(comps)
    n_vars = (C - 1) + sum(comps[c] - 1 for c in range(C))
    
    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    start_time = time.time()
    
    for _ in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization of cuts
        # x cuts should be sorted, y cuts should be sorted
        # We optimize on raw values and sort inside objective? 
        # No, better to use a transformation or just penalize unsorted.
        # Simpler: Initialize sorted, and use bounds to keep them in [0,1].
        # But if they cross, the geometry breaks.
        # Strategy: Optimize on "gaps" or use a monotonic constraint?
        # Easiest: Initialize random sorted cuts.
        # If optimizer crosses them, the objective becomes weird (negative widths?).
        # Let's ensure widths/heights are positive by using max(eps, diff).
        
        x_init = np.sort(np.random.uniform(0, 1, C-1))
        y_inits = []
        for c in range(C):
            n_c = comps[c]
            y_init = np.sort(np.random.uniform(0, 1, n_c - 1))
            y_inits.append(y_init)
            
        params_init = np.concatenate([x_init, np.concatenate(y_inits)])
        
        # Bounds: all cuts in [0, 1]
        bounds = [(0, 1)] * n_vars
        
        # Objective function wrapper
        def obj_func(p):
            # To handle crossing cuts, we can sort them inside the calc
            # But sorting changes the topology.
            # Instead, we penalize if cuts are out of order.
            # Or, we just calculate with the given values and if width < 0, return large penalty.
            
            # Let's use a soft penalty for negative widths/heights
            # Actually, _compute_grid_objective uses np.diff. If p[i] > p[i+1], diff is negative.
            # min(w, h) would be negative.
            # We want to maximize sum of positive mins.
            # So we should penalize negative widths/heights heavily.
            
            # Re-implement objective with penalty for unsorted cuts
            idx = 0
            x_cuts = np.zeros(C-1)
            for i in range(C-1):
                x_cuts[i] = p[idx]
                idx += 1
                
            y_cuts_list = []
            penalty = 0.0
            
            # Check x sorting
            for i in range(C-2):
                if x_cuts[i] >= x_cuts[i+1]:
                    penalty += 1000.0
                    
            for c in range(C):
                n_c = comps[c]
                n_cuts = n_c - 1
                y_cuts = np.zeros(n_cuts)
                for j in range(n_cuts):
                    y_cuts[j] = p[idx]
                    idx += 1
                y_cuts_list.append(y_cuts)
                # Check y sorting
                for j in range(n_cuts - 1):
                    if y_cuts[j] >= y_cuts[j+1]:
                        penalty += 1000.0
                        
            if penalty > 0:
                return penalty
            
            # If sorted, compute objective
            val, _, _ = _compute_grid_objective(p, comps)
            return val # val is negative sum, so we minimize negative sum
            
        res = minimize(obj_func, params_init, method='Nelder-Mead', bounds=bounds, options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success or res.fun < best_score:
            # Check if result is valid (cuts sorted)
            p_opt = res.x
            # Verify sorting
            valid = True
            idx = 0
            x_cuts = p_opt[idx:idx+C-1]
            idx += C-1
            for i in range(C-2):
                if x_cuts[i] >= x_cuts[i+1]:
                    valid = False
                    break
            if valid:
                for c in range(C):
                    n_c = comps[c]
                    n_cuts = n_c - 1
                    y_cuts = p_opt[idx:idx+n_cuts]
                    idx += n_cuts
                    for j in range(n_cuts - 1):
                        if y_cuts[j] >= y_cuts[j+1]:
                            valid = False
                            break
                    if not valid:
                        break
                        
            if valid:
                score, centers, sides = _compute_grid_objective(p_opt, comps)
                score = -score # convert back to positive sum
                if score > best_score:
                    best_score = score
                    best_centers = centers
                    best_sides = sides
                    
    return best_score, best_centers, best_sides

def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 squares in unit square.
    Uses grid-based structural optimization.
    """
    start_time = time.time()
    deadline = 590.0
    
    best_score = 0.0
    best_centers = None
    best_sides = None
    
    # Try different numbers of columns
    for C in range(1, 6): # 1 to 5 columns
        if time.time() > deadline:
            break
            
        for comps in _get_compositions(10, C):
            if time.time() > deadline:
                break
                
            # Skip symmetric duplicates? Compositions are ordered, so (1,9) != (9,1)
            # But geometrically, left-right symmetry means we might duplicate work.
            # It's fast enough, so we don't worry.
            
            score, centers, sides = _optimize_composition(comps, n_restarts=20, time_limit=1.0)
            
            if score > best_score:
                best_score = score
                best_centers = centers
                best_sides = sides
                
    # Fallback if no valid solution found
    if best_centers is None:
        # Simple grid
        centers = np.array([[0.5, 0.5]] * 10)
        sides = np.array([0.1] * 10)
        # This is invalid (overlap), so we need a valid fallback
        # Use the 4x4 grid logic from seed
        k = 4
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(10, side, dtype=np.float64)
        best_score = np.sum(best_sides)
        
    # Final validity check
    if not _check_validity(best_centers, best_sides):
        # If the optimized result is somehow invalid (shouldn't happen with sorted cuts), fallback
        k = 4
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(10, side, dtype=np.float64)
        
    return best_centers, best_sides

# EVOLVE-BLOCK-END
