import numpy as np
from scipy.optimize import minimize
import time


def _compute_volume(positions, n_samples=500000, seed=12345):
    """Compute the union volume using Monte Carlo with the specified seed."""
    N = 8
    i_idx = np.arange(N)
    j_idx = np.arange(N)
    I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
    i_flat = I.flatten()
    j_flat = J.flatten()
    
    base_x = positions[:, 0]
    base_y = positions[:, 1]
    
    # Compute box bounds
    x_right = base_x + (i_flat + 1) / 8.0
    y_right = base_y + (j_flat + 1) / 8.0
    X0 = np.min(base_x)
    X1 = np.max(x_right)
    Y0 = np.min(base_y)
    Y1 = np.max(y_right)
    
    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return 0.0
    
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if each point is inside any tube
    # For efficiency, vectorize over tubes
    # Point (a, b, c) is in tube k if:
    # a >= base_x[k] + c*i_flat[k]/8 and a <= base_x[k] + c*i_flat[k]/8 + 1/8
    # b >= base_y[k] + c*j_flat[k]/8 and b <= base_y[k] + c*j_flat[k]/8 + 1/8
    
    # This is expensive to do fully vectorized (500000 x 64), so use chunking
    inside_any = np.zeros(n_samples, dtype=bool)
    chunk_size = 50000
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        a_c = a[start:end, np.newaxis]  # (chunk, 1)
        b_c = b[start:end, np.newaxis]
        c_c = c[start:end, np.newaxis]
        
        # For each tube, check if point is inside
        # x condition: a_c >= bx + c_c*i/8 and a_c <= bx + c_c*i/8 + 1/8
        x_lo = base_x[np.newaxis, :] + c_c * i_flat[np.newaxis, :] / 8.0
        x_hi = x_lo + 1.0 / 8.0
        y_lo = base_y[np.newaxis, :] + c_c * j_flat[np.newaxis, :] / 8.0
        y_hi = y_lo + 1.0 / 8.0
        
        x_in = (a_c >= x_lo) & (a_c <= x_hi)
        y_in = (b_c >= y_lo) & (b_c <= y_hi)
        tube_in = x_in & y_in  # (chunk, 64)
        
        inside_any[start:end] = np.any(tube_in, axis=1)
    
    fraction = np.sum(inside_any) / n_samples
    return fraction * box_area


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses optimization to minimize the union volume of 64 sheared tubes.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Initial guess: Keich binary construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Compute initial volume
    best_positions = positions.copy()
    best_volume = _compute_volume(positions, n_samples=500000, seed=12345)
    
    # Use Nelder-Mead optimization
    def objective(pos_flat):
        pos = pos_flat.reshape(64, 2)
        return _compute_volume(pos, n_samples=50000, seed=12345)
    
    # Try optimization with limited iterations
    try:
        result = minimize(
            objective,
            positions.flatten(),
            method='Nelder-Mead',
            options={
                'maxiter': 200,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        if result.fun < best_volume:
            best_positions = result.x.reshape(64, 2)
            best_volume = result.fun
    except Exception:
        pass
    
    # If we have time, try more optimization
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:  # Leave 60 seconds buffer
        try:
            # Continue from best position
            result2 = minimize(
                objective,
                best_positions.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-7,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            
            if result2.fun < best_volume:
                best_positions = result2.x.reshape(64, 2)
                best_volume = result2.fun
        except Exception:
            pass
    
    # Final check with full samples
    final_volume = _compute_volume(best_positions, n_samples=500000, seed=12345)
    if final_volume <= 0 or not np.isfinite(final_volume):
        # Fall back to initial
        return positions
    
    return best_positions
