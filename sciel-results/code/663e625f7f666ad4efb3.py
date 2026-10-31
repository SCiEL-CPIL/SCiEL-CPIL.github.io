"""Circle packing in perimeter-4 rectangle: DE global search + Newton local polish."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 21
START = time.time()
DEADLINE = START + 140.0  # Safety margin before 150s kill

def _get_wh(z):
    w = z[-1]
    h = 2.0 - w
    return w, h

def _evaluate_packing(z):
    """
    Evaluate a packing configuration.
    Returns (is_valid, value).
    z = [x1, y1, r1, ..., xN, yN, rN, w]
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w, h = _get_wh(z)
    
    if w < 0.2 or h < 0.2:
        return False, 0.0
    
    # Boundary check
    if np.any(x < 0) or np.any(x > w) or np.any(y < 0) or np.any(y > h):
        return False, 0.0
    if np.any(r < 0):
        return False, 0.0
    if np.any(x - r < -1e-9) or np.any(x + r > w + 1e-9):
        return False, 0.0
    if np.any(y - r < -1e-9) or np.any(y + r > h + 1e-9):
        return False, 0.0
        
    # Pairwise non-overlap check
    # d_ij >= r_i + r_j
    d_mat = np.sqrt((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2)
    r_sum = r[:, None] + r[None, :]
    
    # Ignore diagonal (self-distance 0)
    np.fill_diagonal(d_mat, np.inf)
    
    if np.any(d_mat < r_sum - 1e-9):
        return False, 0.0
        
    return True, float(np.sum(r))

def _newton_polish(z, max_iter=50, tol=1e-10):
    """
    Local refinement using Newton-Raphson on the non-linear system of constraints.
    We treat the 'active' constraints (touching circles/boundaries) as equalities
    and inactive ones as inequalities. This is complex to build rigorously from scratch 
    in short code, so we use a gradient-based local minimization on a penalty 
    function that penalizes overlap and reward growth, but starting from a 
    'tight' configuration found by DE.
    
    Actually, a simpler and more robust 'polish' for this specific task is 
    to run SLSQP with very tight tolerances on the original penalty objective, 
    but the DE start is much better. Let's stick to a fast gradient-free 
    local search or just tight SLSQP.
    """
    # Define a fast penalty objective for local polishing
    def obj(z_local):
        x = z_local[0::3][:N]
        y = z_local[1::3][:N]
        r = z_local[2::3][:N]
        w, h = _get_wh(z_local)
        
        if w < 0.2 or h < 0.2:
            return 1e10
            
        val = -np.sum(r)
        
        # Overlap penalty (quadratic)
        d_mat = np.sqrt((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2)
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(0, r_sum - d_mat)
        val += 1000.0 * np.sum(overlap**2)
        
        # Boundary penalty
        # x > r, x < w-r, y > r, y < h-r
        val += 1000.0 * np.sum(np.maximum(0, r - x)**2)
        val += 1000.0 * np.sum(np.maximum(0, x - (w - r))**2)
        val += 1000.0 * np.sum(np.maximum(0, r - y)**2)
        val += 1000.0 * np.sum(np.maximum(0, y - (h - r))**2)
        
        return val

    # Bounds for local search: keep close to current, but allow some movement
    # Radii can grow slightly, positions can shift
    bounds = []
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w, h = _get_wh(z)
    
    for i in range(N):
        # x bounds
        bounds.append((max(0, x[i] - 0.1), min(w, x[i] + 0.1)))
        # y bounds
        bounds.append((max(0, y[i] - 0.1), min(h, y[i] + 0.1)))
        # r bounds
        bounds.append((max(0.01, r[i] * 0.8), min(0.5, r[i] * 1.2)))
    # w bounds
    bounds.append((max(0.2, w - 0.1), min(1.8, w + 0.1)))
    
    try:
        res = minimize(obj, z, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
        z_polished = res.x
        valid, val = _evaluate_packing(z_polished)
        if valid:
            return z_polished
    except Exception:
        pass
    return z

def _generate_seed(w0, h0):
    """Generate a random initial configuration."""
    z = np.zeros(3 * N + 1)
    # Spread circles randomly
    for i in range(N):
        # Try to place without massive overlap
        x = np.random.uniform(0.1 * w0, 0.9 * w0)
        y = np.random.uniform(0.1 * h0, 0.9 * h0)
        # Estimate radius based on local density, keep small
        r = np.random.uniform(0.05, 0.1)
        z[3*i] = x
        z[3*i+1] = y
        z[3*i+2] = r
    z[-1] = w0
    return z

def construct():
    best_z = None
    best_val = 0.0
    
    # Time budget for DE
    de_budget = 60.0
    polish_budget = 40.0
    
    # Define bounds for DE
    # w in [0.5, 1.5] to avoid degenerate rectangles
    # x, y, r depend on w, but DE handles variable dimensions poorly if bounds are fixed.
    # We must use dynamic bounds or normalize. 
    # Simpler approach: Fix aspect ratio candidates and run DE for each? 
    # Or use a transformation. 
    # Given time constraints, we'll run DE on a normalized space or specific aspect ratios.
    # Let's pick a few aspect ratios and run DE for each.
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2]
    
    for w0 in aspect_ratios:
        if time.time() > DEADLINE - polish_budget - 10:
            break
            
        h0 = 2.0 - w0
        if h0 < 0.4:
            continue
            
        # Bounds for DE
        # x in [0, w0], y in [0, h0], r in [0.01, 0.3]
        bounds = []
        for i in range(N):
            bounds.append((0.0, w0))
            bounds.append((0.0, h0))
            bounds.append((0.01, 0.3))
        bounds.append((w0 - 0.1, w0 + 0.1)) # Optimize w slightly around w0
        
        # Objective for DE: Maximize sum(r) -> Minimize -sum(r) with heavy penalty for invalidity
        def de_objective(z):
            x = z[0::3][:N]
            y = z[1::3][:N]
            r = z[2::3][:N]
            w, h = _get_wh(z)
            
            if w < 0.2 or h < 0.2:
                return 1e10
                
            # Clip positions to box to help DE
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            
            # Overlap
            d_mat = np.sqrt((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2)
            r_sum = r[:, None] + r[None, :]
            overlap = np.maximum(0, r_sum - d_mat)
            pen = np.sum(overlap**2)
            
            # Boundary
            b_pen = np.sum(np.maximum(0, r - x)**2) + np.sum(np.maximum(0, x - (w - r))**2)
            b_pen += np.sum(np.maximum(0, r - y)**2) + np.sum(np.maximum(0, y - (h - r))**2)
            
            val = -np.sum(r) + 500.0 * pen + 500.0 * b_pen
            return val
            
        try:
            # Run DE with limited iterations to respect time
            max_iter = 50
            seed = int(time.time() * 1000) % 10000
            
            res = differential_evolution(
                de_objective, 
                bounds=bounds, 
                maxiter=max_iter, 
                popsize=10, 
                tol=1e-4, 
                seed=seed,
                polish=False # We will polish separately
            )
            
            z_candidate = res.x
            valid, val = _evaluate_packing(z_candidate)
            
            if valid and val > best_val:
                # Polish
                z_polished = _newton_polish(z_candidate)
                valid_p, val_p = _evaluate_packing(z_polished)
                if valid_p and val_p > best_val:
                    best_val = val_p
                    best_z = z_polished
                elif valid and val > best_val:
                    best_val = val
                    best_z = z_candidate
                    
        except Exception as e:
            continue

    # If we still have time, do a global SLSQP refinement on the best found
    if best_z is not None and time.time() < DEADLINE - 5:
        def final_obj(z):
            x = z[0::3][:N]
            y = z[1::3][:N]
            r = z[2::3][:N]
            w, h = _get_wh(z)
            if w < 0.2 or h < 0.2: return 1e10
            d_mat = np.sqrt((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2)
            r_sum = r[:, None] + r[None, :]
            overlap = np.maximum(0, r_sum - d_mat)
            b_pen = np.sum(np.maximum(0, r - x)**2) + np.sum(np.maximum(0, x - (w - r))**2)
            b_pen += np.sum(np.maximum(0, r - y)**2) + np.sum(np.maximum(0, y - (h - r))**2)
            return -np.sum(r) + 1000.0 * (np.sum(overlap**2) + b_pen)
            
        bounds_final = [(0, 2)] * (3*N) + [(0.3, 1.7)]
        try:
            res = minimize(final_obj, best_z, method='SLSQP', bounds=bounds_final, options={'maxiter': 100, 'ftol': 1e-12})
            z_final = res.x
            valid, val = _evaluate_packing(z_final)
            if valid and val > best_val:
                best_val = val
                best_z = z_final
        except Exception:
            pass

    # Construct return array
    if best_z is None:
        # Fallback: Uniform grid
        side = int(np.ceil(np.sqrt(N)))
        z = np.zeros(3*N + 1)
        w = 1.0
        h = 1.0
        z[-1] = w
        for i in range(N):
            row, col = divmod(i, side)
            z[3*i] = (col + 0.5) / side * w
            z[3*i+1] = (row + 0.5) / side * h
            z[3*i+2] = 0.45 / side
        # Ensure validity
        valid, val = _evaluate_packing(z)
        if not valid:
            # Shrink radii
            r_idx = np.arange(2, 3*N, 3)
            z[r_idx] *= 0.8
            valid, val = _evaluate_packing(z)
        
        best_z = z
        best_val = val if valid else 0.0

    x = best_z[0::3][:N]
    y = best_z[1::3][:N]
    r = best_z[2::3][:N]
    
    # Safety: slight shrink to ensure strict disjointness in floating point
    r = r * (1 - 1e-8)
    
    return np.column_stack((x, y, r))

# Main entry point
if __name__ == "__main__":
    result = construct()
    print(result)
