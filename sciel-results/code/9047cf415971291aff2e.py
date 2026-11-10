# EVOLVE-BLOCK-START
import numpy as np
import time


def _mc_volume_fast(positions, n_samples=50000):
    """Fast Monte Carlo volume estimate with fewer samples for speed."""
    N = 8
    n = len(positions)
    i_idx = np.arange(n) // N
    j_idx = np.arange(n) % N
    
    x0 = positions[:, 0]
    y0 = positions[:, 1]
    
    X0 = np.min(x0)
    X1 = np.max(x0 + (i_idx + 1) / N)
    Y0 = np.min(y0)
    Y1 = np.max(y0 + (j_idx + 1) / N)
    
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return float('inf')
    
    rng = np.random.default_rng(42)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if any point is inside at least one tube
    inside = np.zeros(n_samples, dtype=bool)
    for k in range(n):
        i = i_idx[k]
        j = j_idx[k]
        x_lo = x0[k] + c * i / N
        x_hi = x_lo + 1.0 / N
        y_lo = y0[k] + c * j / N
        y_hi = y_lo + 1.0 / N
        inside |= (a >= x_lo) & (a <= x_hi) & (b >= y_lo) & (b <= y_hi)
    
    frac = np.mean(inside)
    return frac * (X1 - X0) * (Y1 - Y0)


def _keich_positions_1d(N):
    """Generate Keich binary construction positions for 1D."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _local_search_2d(positions, max_iter=500, step_size=0.05):
    """Local search to minimize 2D union volume."""
    best_positions = positions.copy()
    best_vol = _mc_volume_fast(best_positions)
    
    deadline = time.time() + 580  # Leave margin for evaluation
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try perturbing each position
        for k in range(len(positions)):
            for dim in [0, 1]:
                for delta in [step_size, -step_size]:
                    trial = best_positions.copy()
                    trial[k, dim] += delta
                    
                    vol = _mc_volume_fast(trial)
                    if vol < best_vol:
                        best_positions = trial
                        best_vol = vol
                        improved = True
        
        if not improved:
            # Try smaller steps
            step_size *= 0.5
            if step_size < 0.001:
                break
    
    return best_positions, best_vol


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses Keich binary construction as initialization, then local search to
    minimize the 2D union volume.
    
    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order.
    """
    N = 8
    start_time = time.time()
    deadline = time.time() + 600  # Hard deadline
    
    # Initial Keich construction
    keich_1d = _keich_positions_1d(N)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = keich_1d[i_flat]
    base_y = keich_1d[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 580:
        return positions
    
    # Local search to improve
    improved_positions, improved_vol = _local_search_2d(positions, max_iter=200, step_size=0.02)
    
    # Check if improvement is valid
    if np.all(np.isfinite(improved_positions)) and improved_vol > 0:
        return improved_positions
    else:
        return positions


# EVOLVE-BLOCK-END
