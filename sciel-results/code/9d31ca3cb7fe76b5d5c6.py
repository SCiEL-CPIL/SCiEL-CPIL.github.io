# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _calc_area(x):
    """
    Compute the area of the union of the 64 triangles for given positions x.
    n = 64
    """
    n = 64
    w = 1.0 / n
    
    # a_i = (i + 1) / 64
    a = (np.arange(n) + 1) / n
    
    # y_m = (m + 0.5) / 20000, m = 0..19999
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    
    # L_i = x_i + a_i * y
    # Shape: (n, 20000)
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L_i for each y_m
    L_sorted = np.sort(L, axis=0)
    
    diffs = np.diff(L_sorted, axis=0)
    added_lengths = np.minimum(diffs, w)
    
    union_lengths = w + np.sum(added_lengths, axis=0)
    
    area = np.mean(union_lengths)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimize the base positions x for the Kakeya needle problem (n=64).
    Strategy:
    1. Start with Keich construction.
    2. Use L-BFGS-B with a fast vectorized objective for global-ish convergence.
    3. Use a pairwise refinement heuristic to escape local minima.
    """
    n = 64
    k = int(np.log2(n))
    
    # 1. Generate Keich initial guess
    j = np.arange(1, k + 1)
    keich_positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        keich_positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(keich_positions, dtype=np.float64)
    
    start_time = time.time()
    timeout = 600.0
    margin = 10.0
    
    best_x = x0.copy()
    best_area = _calc_area(x0)
    
    # 2. L-BFGS-B Optimization
    # L-BFGS-B is much faster than Nelder-Mead for smooth-ish functions and handles bounds well.
    # The area function is continuous but has kinks where L_i = L_j or L_{j+1} - L_j = w.
    # However, L-BFGS-B often navigates these well.
    
    def objective(x):
        if time.time() - start_time > timeout - margin:
            raise TimeoutError
        return _calc_area(x)
    
    # We don't have strict bounds, but keeping x in a reasonable range helps.
    # The Keich construction has x in [0, 1).
    # Let's allow a bit of slack.
    bounds = [(-0.5, 1.5)] * n
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'maxls': 50, 'maxcor': 20})
        if res.success or res.fun < best_area:
            if np.all(np.isfinite(res.x)):
                best_x = res.x
                best_area = res.fun
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # 3. Pairwise Refinement Heuristic
    # The area is sum of min(gaps, w). 
    # To minimize area, we want gaps to be small (overlap) or exactly w (no extra length).
    # Actually, the formula is w + sum min(L_{j+1} - L_j, w).
    # The total span is fixed by the outermost intervals? No.
    # Let's look at the structure.
    # If we sort the triangles by their position at a specific y, the union length depends on the gaps.
    # A local search that swaps or shifts pairs of triangles that are "close" might help.
    
    # A more effective heuristic for this specific "union of intervals" problem:
    # The area is minimized when the intervals overlap as much as possible, 
    # but the "width" of each interval is fixed at w.
    # The term is w + sum min(gap, w).
    # If gap < w, the term is gap. If gap > w, the term is w.
    # So we want to minimize the sum of gaps that are < w, and keep gaps >= w as they are (contribute w).
    # Wait, if gap >= w, it contributes w. If gap < w, it contributes gap.
    # So the contribution is always min(gap, w).
    # To minimize the sum, we want gaps to be as small as possible.
    # This implies we want the intervals to be "clumped" together.
    # However, the intervals move relative to each other as y changes.
    # L_i(y) = x_i + a_i y.
    # Relative velocity v_ij = a_i - a_j.
    # If v_ij = 0, they move together.
    # If v_ij != 0, they separate or converge.
    
    # Let's try a simple coordinate descent / local search with small steps.
    # Randomly pick an index i, and try to shift x_i slightly to reduce area.
    
    rng = np.random.default_rng(42)
    
    # Precompute a and y for speed in the loop
    a = (np.arange(n) + 1) / n
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    w = 1.0 / n
    
    def _calc_area_fast(x):
        L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
        L_sorted = np.sort(L, axis=0)
        diffs = np.diff(L_sorted, axis=0)
        added_lengths = np.minimum(diffs, w)
        union_lengths = w + np.sum(added_lengths, axis=0)
        return np.mean(union_lengths)
    
    current_x = best_x.copy()
    current_area = best_area
    
    step_size = 0.01
    no_improve_count = 0
    
    while time.time() - start_time < timeout - margin:
        # Randomly select a few indices to perturb
        indices = rng.choice(n, size=4, replace=False)
        
        improved = False
        for idx in indices:
            # Try small positive and negative shifts
            for delta in [step_size, -step_size, step_size*0.1, -step_size*0.1]:
                trial_x = current_x.copy()
                trial_x[idx] += delta
                trial_area = _calc_area_fast(trial_x)
                
                if trial_area < current_area - 1e-10:
                    current_x = trial_x
                    current_area = trial_area
                    improved = True
                    break # Move to next index if improved
            
            if not improved:
                # If no index in this batch improved, we might be stuck.
                # Shrink step size occasionally? Or just continue.
                pass
        
        if time.time() - start_time > timeout - margin:
            break
            
        if np.all(np.isfinite(current_x)) and current_area < best_area:
            best_x = current_x.copy()
            best_area = current_area
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 500:
                # Reset to best and try a larger perturbation?
                # Or just break if stuck for too long.
                # Let's try a random restart from best_x with larger noise
                current_x = best_x + rng.normal(0, 0.05, n)
                current_area = _calc_area_fast(current_x)
                no_improve_count = 0
                # If this restart is worse, go back to best
                if current_area > best_area:
                    current_x = best_x.copy()
                    current_area = best_area

    return best_x

# EVOLVE-BLOCK-END
