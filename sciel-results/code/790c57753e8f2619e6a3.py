# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _union_volume_fast(params, N=8, n_c=20, n_xy=20):
    """
    Fast approximation of union volume using Gauss-Legendre quadrature.
    params: shape (64, 2) array of (x_k, y_k) base positions.
    Returns: estimated union volume.
    """
    pos = params.reshape(-1, 2)
    x = pos[:, 0]
    y = pos[:, 1]
    i_idx = np.arange(N * N) // N
    j_idx = np.arange(N * N) % N
    
    # Compute bounding box
    X0 = np.min(x)
    X1 = np.max(x + (i_idx + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_idx + 1) / N)
    
    area = (X1 - X0) * (Y1 - Y0)
    if area <= 1e-9:
        return 1e9  # Invalid, return large volume
    
    # Gauss-Legendre nodes and weights for [0, 1]
    c_nodes, c_weights = np.polynomial.legendre.leggauss(n_c)
    c_nodes = 0.5 * (c_nodes + 1)  # Map to [0, 1]
    c_weights = 0.5 * c_weights
    
    xy_nodes, xy_weights = np.polynomial.legendre.leggauss(n_xy)
    xy_nodes = 0.5 * (xy_nodes + 1)  # Map to [0, 1]
    xy_weights = 0.5 * xy_weights
    
    # Total volume = area * integral over c of (fraction of xy area covered)
    # We integrate over c using quadrature, and for each c, compute the 2D union area
    total_volume = 0.0
    for ci in range(n_c):
        c = c_nodes[ci]
        w_c = c_weights[ci]
        
        # For this c, compute the x and y intervals for each tube
        # Tube k: x in [x_k + c*i_k/N, x_k + c*i_k/N + 1/N]
        #          y in [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
        x_starts = x + c * i_idx / N
        y_starts = y + c * j_idx / N
        x_ends = x_starts + 1.0 / N
        y_ends = y_starts + 1.0 / N
        
        # For 2D union area, we need to compute the area covered by the union of rectangles
        # This is a classic problem. We can use a sweep-line or grid-based approximation.
        # For speed, use a coarse grid within the bounding box.
        # But for better accuracy, let's use an exact-ish approach with interval merging in 1D for each y-slice.
        
        # Actually, for 64 rectangles, we can use a more efficient method:
        # Discretize the y-range into segments determined by all y_starts and y_ends.
        # For each segment, the set of active x-intervals is constant, so we can merge them.
        
        all_y = np.concatenate([y_starts, y_ends])
        all_y = np.unique(all_y)
        
        if len(all_y) < 2:
            continue
        
        # For each y-segment, find which tubes are active
        # and compute the union of their x-intervals
        y_area = 0.0
        for seg_i in range(len(all_y) - 1):
            y_lo = all_y[seg_i]
            y_hi = all_y[seg_i + 1]
            dy = y_hi - y_lo
            if dy <= 0:
                continue
            
            # Find tubes that cover this y-segment (at least partially)
            # A tube covers y-segment if y_start <= y_hi and y_end >= y_lo
            active = (y_starts <= y_hi) & (y_ends >= y_lo)
            
            if not np.any(active):
                continue
            
            # Get x-intervals for active tubes
            x_s = x_starts[active]
            x_e = x_ends[active]
            
            # Merge x-intervals
            intervals = np.stack([x_s, x_e], axis=1)
            intervals = intervals[np.argsort(intervals[:, 0])]
            
            merged = []
            cur_start, cur_end = intervals[0]
            for s, e in intervals[1:]:
                if s <= cur_end:
                    cur_end = max(cur_end, e)
                else:
                    merged.append((cur_start, cur_end))
                    cur_start, cur_end = s, e
            merged.append((cur_start, cur_end))
            
            x_union_len = sum(e - s for s, e in merged)
            y_area += x_union_len * dy
        
        total_volume += w_c * y_area
    
    return area * total_volume


def _objective(params):
    """Objective function for optimization."""
    N = 8
    pos = params.reshape(N * N, 2)
    return _union_volume_fast(pos, N=N, n_c=10, n_xy=10)


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    n_tubes = N * N
    deadline = time.time() + 620  # Safety margin
    
    # Initial guess: Keich construction
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    best_params = np.stack([base_x, base_y], axis=1).flatten().copy()
    best_vol = _objective(best_params)
    
    # Try differential evolution with time limit
    try:
        def time_check():
            return time.time() > deadline - 30
        
        # Bounds: x and y can range roughly from -1 to 2
        bounds = [(-1.0, 2.0)] * (2 * n_tubes)
        
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=50,
            popsize=15,
            seed=12345,
            tol=1e-6,
            polish=True,
            callback=lambda xk, convergence: time_check() and 1 or None
        )
        
        if result.fun < best_vol:
            best_vol = result.fun
            best_params = result.x.copy()
    except Exception:
        pass
    
    # If time remains, try local optimization from best point
    if time.time() < deadline - 60:
        try:
            result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7}
            )
            if result.fun < best_vol:
                best_vol = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Convert back to (64, 2) shape
    positions = best_params.reshape(n_tubes, 2)
    
    # Validate
    if not np.all(np.isfinite(positions)):
        # Return initial guess
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
