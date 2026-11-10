# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _exact_1d_union_length(intervals):
    """
    Compute the exact measure (total length) of the union of 1D intervals.
    
    Args:
        intervals: array of shape (n, 2) where intervals[k] = [lo_k, hi_k]
    
    Returns:
        Total length of the union of all intervals.
    """
    n = intervals.shape[0]
    if n == 0:
        return 0.0
    
    lo = intervals[:, 0]
    hi = intervals[:, 1]
    
    # Sort by lower endpoint
    order = np.argsort(lo, kind='stable')
    lo_s = lo[order]
    hi_s = hi[order]
    
    total = 0.0
    current_lo = lo_s[0]
    current_hi = hi_s[0]
    
    for k in range(1, n):
        if lo_s[k] > current_hi:
            # Disjoint: add current segment
            total += current_hi - current_lo
            current_lo = lo_s[k]
            current_hi = hi_s[k]
        else:
            # Overlapping: extend current_hi if needed
            if hi_s[k] > current_hi:
                current_hi = hi_s[k]
    
    # Add last segment
    total += current_hi - current_lo
    return total


def _cross_section_area_2d(base_x, base_y, i_flat, j_flat, c, N=8):
    """
    Compute the 2D cross-section area of the union at height c.
    
    For each tube k with base (x_k, y_k) and indices (i_k, j_k):
        x-interval: [x_k + c*i_k/N, x_k + c*i_k/N + 1/N]
        y-interval: [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    
    The union area is the measure of the union of axis-aligned rectangles.
    Since x-intervals depend only on (x_k, i_k) and y-intervals only on (y_k, j_k),
    and the rectangles are products of x-intervals and y-intervals, the union area
    equals the product of the 1D x-union length and 1D y-union length.
    
    Args:
        base_x: array of shape (64,), base x positions
        base_y: array of shape (64,), base y positions
        i_flat: array of shape (64,), i indices (k // 8)
        j_flat: array of shape (64,), j indices (k % 8)
        c: scalar, height in [0, 1]
        N: grid parameter, default 8
    
    Returns:
        2D cross-section area at height c
    """
    N = float(N)
    shift_x = c * i_flat / N
    shift_y = c * j_flat / N
    
    x_lo = base_x + shift_x
    x_hi = x_lo + 1.0 / N
    y_lo = base_y + shift_y
    y_hi = y_lo + 1.0 / N
    
    x_intervals = np.stack([x_lo, x_hi], axis=1)
    y_intervals = np.stack([y_lo, y_hi], axis=1)
    
    x_len = _exact_1d_union_length(x_intervals)
    y_len = _exact_1d_union_length(y_intervals)
    
    return x_len * y_len


def _union_volume(base_x, base_y, i_flat, j_flat, N=8, n_quad=64):
    """
    Estimate the union volume via Gauss-Legendre quadrature over c in [0, 1].
    
    Args:
        base_x: array of shape (64,)
        base_y: array of shape (64,)
        i_flat: array of shape (64,)
        j_flat: array of shape (64,)
        N: grid parameter
        n_quad: number of quadrature points
    
    Returns:
        Estimated union volume
    """
    c_nodes, c_weights = np.polynomial.legendre.leggauss(n_quad)
    # Map from [-1, 1] to [0, 1]
    c_nodes = 0.5 * (c_nodes + 1.0)
    c_weights = 0.5 * c_weights
    
    areas = np.array([_cross_section_area_2d(base_x, base_y, i_flat, j_flat, c, N) for c in c_nodes])
    return np.sum(c_weights * areas)


def _objective(params, i_flat, j_flat, N=8, n_quad=32):
    """
    Objective function for optimization: union volume.
    
    Args:
        params: array of shape (128,), [base_x (64), base_y (64)]
    
    Returns:
        Union volume (scalar)
    """
    base_x = params[:64]
    base_y = params[64:]
    return _union_volume(base_x, base_y, i_flat, j_flat, N, n_quad)


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Minimizes the union volume of 64 sheared tubes using gradient-based optimization
    with random restarts and a time budget.
    
    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order,
            column 0 = base x-coordinate, column 1 = base y-coordinate.
    """
    N = 8
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    deadline = time.time() + 600  # 600 second deadline
    
    # Initial best: use the Keich construction as a starting point
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    
    base_x0 = keich[i_flat]
    base_y0 = keich[j_flat]
    params0 = np.concatenate([base_x0, base_y0])
    
    best_params = params0.copy()
    best_vol = _union_volume(base_x0, base_y0, i_flat, j_flat, N, n_quad=64)
    
    # Try multiple random restarts with L-BFGS-B optimization
    rng = np.random.default_rng(42)
    n_restarts = 0
    
    while time.time() < deadline - 60:  # Leave 60s margin
        n_restarts += 1
        
        # Generate initial guess: perturb the Keich solution
        if n_restarts == 1:
            x0 = params0.copy()
        else:
            # Random perturbation around the Keich solution
            scale = 0.5
            x0 = params0 + scale * rng.standard_normal(128)
        
        # Optimize using L-BFGS-B
        result = minimize(
            _objective,
            x0,
            args=(i_flat, j_flat, N, 32),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < best_vol:
            best_params = result.x
            best_vol = result.fun
        
        # Also try a few more random starting points if time permits
        for _ in range(3):
            if time.time() > deadline - 60:
                break
            x0_rand = params0 + 0.3 * rng.standard_normal(128)
            result = minimize(
                _objective,
                x0_rand,
                args=(i_flat, j_flat, N, 32),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_vol:
                best_params = result.x
                best_vol = result.fun
    
    # Final: extract best positions
    base_x = best_params[:64]
    base_y = best_params[64:]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
