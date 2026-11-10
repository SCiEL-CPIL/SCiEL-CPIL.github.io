# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _union_area_2d(x_intervals, y_intervals):
    """
    Compute the area of the union of axis-aligned rectangles.
    x_intervals: list of (x0, x1)
    y_intervals: list of (y0, y1)
    Each rectangle is x_intervals[i] x y_intervals[i].
    """
    n = len(x_intervals)
    if n == 0:
        return 0.0
    
    # Collect all x-coordinates
    x_coords = []
    for x0, x1 in x_intervals:
        x_coords.append(x0)
        x_coords.append(x1)
    x_coords = sorted(set(x_coords))
    
    if len(x_coords) < 2:
        return 0.0
    
    # For each strip between consecutive x-coords, compute the union of y-intervals
    total_area = 0.0
    for s in range(len(x_coords) - 1):
        x_lo = x_coords[s]
        x_hi = x_coords[s + 1]
        dx = x_hi - x_lo
        if dx <= 0:
            continue
        
        # Find which rectangles cover this strip
        y_intervals_in_strip = []
        for i in range(n):
            x0, x1 = x_intervals[i]
            y0, y1 = y_intervals[i]
            # Rectangle i covers [x_lo, x_hi] if x0 <= x_lo and x1 >= x_hi
            # Actually, we need the rectangle to overlap with (x_lo, x_hi)
            if x0 < x_hi and x1 > x_lo:
                y_intervals_in_strip.append((y0, y1))
        
        if not y_intervals_in_strip:
            continue
        
        # Compute union length of y_intervals_in_strip
        y_intervals_in_strip.sort()
        merged = []
        cur_y0, cur_y1 = y_intervals_in_strip[0]
        for y0, y1 in y_intervals_in_strip[1:]:
            if y0 <= cur_y1:
                cur_y1 = max(cur_y1, y1)
            else:
                merged.append((cur_y0, cur_y1))
                cur_y0, cur_y1 = y0, y1
        merged.append((cur_y0, cur_y1))
        
        y_union_len = sum(y1 - y0 for y0, y1 in merged)
        total_area += dx * y_union_len
    
    return total_area


def _compute_volume_exact(base_x, base_y, n_c_samples=200):
    """
    Compute the union volume by integrating the 2D cross-sectional area over c in [0,1].
    base_x: array of shape (8,) - base x positions for i=0..7
    base_y: array of shape (8,) - base y positions for j=0..7
    """
    N = 8
    # Use Gauss-Legendre quadrature for better accuracy
    c_vals, weights = np.polynomial.legendre.leggauss(n_c_samples)
    c_vals = 0.5 * (c_vals + 1)  # map from [-1,1] to [0,1]
    weights = 0.5 * weights
    
    total_vol = 0.0
    for c, w in zip(c_vals, weights):
        x_intervals = []
        y_intervals = []
        for i in range(N):
            for j in range(N):
                x0 = base_x[i] + c * i / N
                x1 = x0 + 1.0 / N
                y0 = base_y[j] + c * j / N
                y1 = y0 + 1.0 / N
                x_intervals.append((x0, x1))
                y_intervals.append((y0, y1))
        
        area = _union_area_2d(x_intervals, y_intervals)
        total_vol += w * area
    
    return total_vol


def _objective(params):
    """
    Objective function for optimization.
    params: array of shape (16,) = [base_x(0),...,base_x(7), base_y(0),...,base_y(7)]
    """
    base_x = params[:8]
    base_y = params[8:]
    return _compute_volume_exact(base_x, base_y, n_c_samples=100)


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 580  # leave 20s margin
    
    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    init_params = np.concatenate([keich, keich])
    
    # Try L-BFGS-B optimization
    best_params = init_params.copy()
    best_vol = _compute_volume_exact(best_params[:8], best_params[8:], n_c_samples=200)
    
    try:
        result = minimize(
            _objective,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success or result.fun < best_vol:
            best_params = result.x
            best_vol = result.fun
        
        # Check time
        if time.time() > deadline - 60:
            pass  # skip refinement
        else:
            # Refine with more quadrature points
            result2 = minimize(
                lambda p: _compute_volume_exact(p[:8], p[8:], n_c_samples=200),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result2.fun < best_vol:
                best_params = result2.x
                best_vol = result2.fun
    except Exception:
        pass
    
    # Check time again
    if time.time() > deadline - 30:
        pass
    else:
        # Try a few random restarts
        rng = np.random.default_rng(42)
        for restart in range(5):
            if time.time() > deadline - 30:
                break
            random_params = rng.uniform(-0.5, 0.5, 16)
            try:
                result_r = minimize(
                    _objective,
                    random_params,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result_r.fun < best_vol:
                    best_params = result_r.x
                    best_vol = result_r.fun
            except Exception:
                pass
    
    base_x = best_params[:8]
    base_y = best_params[8:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    positions = np.stack([base_x[i_flat], base_y[j_flat]], axis=1)
    return positions


# EVOLVE-BLOCK-END
