# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _union_length(intervals):
    """
    Compute the total length of the union of 1D intervals.
    intervals: array of shape (n, 2) with columns [start, end]
    Returns: total length of union
    """
    if len(intervals) == 0:
        return 0.0
    intervals = np.sort(intervals, axis=0)
    # Merge overlapping intervals
    merged = np.empty_like(intervals)
    merged[0] = intervals[0]
    count = 1
    for i in range(1, len(intervals)):
        if intervals[i, 0] <= merged[count-1, 1]:
            merged[count-1, 1] = max(merged[count-1, 1], intervals[i, 1])
        else:
            merged[count] = intervals[i]
            count += 1
    merged = merged[:count]
    return np.sum(merged[:, 1] - merged[:, 0])


def _compute_volume_exact(positions, N=8, n_samples=50):
    """
    Compute the exact volume using numerical integration over c.
    For each c, compute the 1D union lengths in x and y, then integrate.
    """
    N = 8
    half = 1.0 / N
    i_idx = np.arange(N)
    j_idx = np.arange(N)
    
    # positions is (64, 2), row k has i=k//N, j=k%N
    base_x = positions[:, 0].copy()
    base_y = positions[:, 1].copy()
    
    # For a given c, x-interval for tube (i,j) is [x_k + c*i/N, x_k + c*i/N + 1/N]
    # y-interval is [y_k + c*j/N, y_k + c*j/N + 1/N]
    
    # Use Gauss-Legendre quadrature for accurate integration
    c_points, c_weights = np.polynomial.legendre.leggauss(n_samples)
    c_points = 0.5 * (c_points + 1)  # map from [-1,1] to [0,1]
    c_weights = 0.5 * c_weights
    
    total_vol = 0.0
    for c, w in zip(c_points, c_weights):
        # Compute x-intervals for all 64 tubes
        x_starts = base_x + c * (np.arange(64) // N) / N
        x_ends = x_starts + half
        x_intervals = np.column_stack([x_starts, x_ends])
        x_len = _union_length(x_intervals)
        
        # Compute y-intervals
        y_starts = base_y + c * (np.arange(64) % N) / N
        y_ends = y_starts + half
        y_intervals = np.column_stack([y_starts, y_ends])
        y_len = _union_length(y_intervals)
        
        total_vol += w * x_len * y_len
    
    return total_vol


def _objective_and_grad(pos_flat, N=8):
    """
    Objective function and gradient for L-BFGS-B.
    pos_flat: 1D array of length 128 (64 x-coords, 64 y-coords)
    """
    positions = pos_flat.reshape(64, 2)
    
    # Compute volume
    vol = _compute_volume_exact(positions, N=N, n_samples=30)
    
    # Numerical gradient
    eps = 1e-7
    grad = np.zeros_like(pos_flat)
    for idx in range(len(pos_flat)):
        pos_plus = pos_flat.copy()
        pos_plus[idx] += eps
        vol_plus = _compute_volume_exact(pos_plus.reshape(64, 2), N=N, n_samples=15)
        
        pos_minus = pos_flat.copy()
        pos_minus[idx] -= eps
        vol_minus = _compute_volume_exact(pos_minus.reshape(64, 2), N=N, n_samples=15)
        
        grad[idx] = (vol_plus - vol_minus) / (2 * eps)
    
    return vol, grad


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with N=8.
    Returns (64, 2) array of base positions.
    """
    N = 8
    half = 1.0 / N
    start_time = time.time()
    deadline = 600.0  # seconds
    
    # Initialize with Keich construction
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
    
    base_x_init = keich[i_flat]
    base_y_init = keich[j_flat]
    positions_init = np.stack([base_x_init, base_y_init], axis=1)
    pos_flat_init = positions_init.flatten()
    
    best_positions = positions_init.copy()
    best_vol = _compute_volume_exact(best_positions, N=N, n_samples=50)
    
    # Try L-BFGS-B optimization
    try:
        def obj_func(x):
            vol, _ = _objective_and_grad(x, N=N)
            return vol
        
        result = minimize(
            obj_func,
            pos_flat_init,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 2000}
        )
        
        if result.success and result.fun < best_vol:
            best_positions = result.x.reshape(64, 2)
            best_vol = result.fun
            print(f"L-BFGS-B improved: vol={best_vol:.6f}")
    except Exception as e:
        print(f"L-BFGS-B failed: {e}")
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 60:
        return best_positions
    
    # Try coordinate descent as a refinement
    try:
        positions_cd = best_positions.copy()
        for iteration in range(50):
            if time.time() - start_time > deadline - 60:
                break
            improved = False
            for k in range(64):
                for coord in range(2):
                    # Try small perturbations
                    for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                        pos_test = positions_cd.copy()
                        pos_test[k, coord] += delta
                        vol_test = _compute_volume_exact(pos_test, N=N, n_samples=20)
                        if vol_test < _compute_volume_exact(positions_cd, N=N, n_samples=20):
                            positions_cd[k, coord] += delta
                            improved = True
                            break
            if not improved:
                break
        
        vol_cd = _compute_volume_exact(positions_cd, N=N, n_samples=50)
        if vol_cd < best_vol:
            best_positions = positions_cd
            best_vol = vol_cd
            print(f"Coordinate descent improved: vol={best_vol:.6f}")
    except Exception as e:
        print(f"Coordinate descent failed: {e}")
    
    # Final polish with smaller steps
    try:
        positions_polish = best_positions.copy()
        for iteration in range(30):
            if time.time() - start_time > deadline - 30:
                break
            improved = False
            for k in range(64):
                for coord in range(2):
                    for delta in [0.001, -0.001, 0.0005, -0.0005, 0.0001, -0.0001]:
                        pos_test = positions_polish.copy()
                        pos_test[k, coord] += delta
                        vol_test = _compute_volume_exact(pos_test, N=N, n_samples=15)
                        vol_curr = _compute_volume_exact(positions_polish, N=N, n_samples=15)
                        if vol_test < vol_curr:
                            positions_polish[k, coord] += delta
                            improved = True
                            break
            if not improved:
                break
        
        vol_polish = _compute_volume_exact(positions_polish, N=N, n_samples=50)
        if vol_polish < best_vol:
            best_positions = positions_polish
            best_vol = vol_polish
            print(f"Polish improved: vol={best_vol:.6f}")
    except Exception as e:
        print(f"Polish failed: {e}")
    
    return best_positions


# EVOLVE-BLOCK-END
