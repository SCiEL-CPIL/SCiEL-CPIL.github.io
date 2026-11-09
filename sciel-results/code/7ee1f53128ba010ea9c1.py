import numpy as np
import time


def _compute_volume(positions, seed=12345, n_samples=500000):
    """Compute the union volume using the exact Monte Carlo method from the task."""
    N = 8
    x = positions[:, 0]
    y = positions[:, 1]
    
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    X0 = np.min(x)
    X1 = np.max(x + (i_idx + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_idx + 1) / N)
    
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return float('inf')
    
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    inside = np.zeros(n_samples, dtype=bool)
    
    for k in range(64):
        i_k = i_idx[k]
        j_k = j_idx[k]
        x_k = x[k]
        y_k = y[k]
        
        x_left = x_k + c * i_k / N
        x_right = x_left + 1.0 / N
        y_left = y_k + c * j_k / N
        y_right = y_left + 1.0 / N
        
        in_x = (a >= x_left) & (a <= x_right)
        in_y = (b >= y_left) & (b <= y_right)
        inside |= (in_x & in_y)
    
    fraction = np.mean(inside)
    V = fraction * (X1 - X0) * (Y1 - Y0)
    return V


def _compute_volume_fast(positions, n_samples=50000):
    """Faster volume estimate using vectorized operations for local search."""
    N = 8
    x = positions[:, 0]
    y = positions[:, 1]
    
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    X0 = np.min(x)
    X1 = np.max(x + (i_idx + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_idx + 1) / N)
    
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return float('inf')
    
    # Use a fixed seed for reproducibility in local search
    rng = np.random.default_rng(999)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Vectorize: for each sample point, check all 64 tubes
    # Shape: (n_samples, 64)
    c_exp = c[:, np.newaxis]  # (n_samples, 1)
    i_exp = i_idx[np.newaxis, :]  # (1, 64)
    j_exp = j_idx[np.newaxis, :]  # (1, 64)
    x_exp = x[np.newaxis, :]  # (1, 64)
    y_exp = y[np.newaxis, :]  # (1, 64)
    
    x_left = x_exp + c_exp * i_exp / N  # (n_samples, 64)
    x_right = x_left + 1.0 / N
    y_left = y_exp + c_exp * j_exp / N
    y_right = y_left + 1.0 / N
    
    in_x = (a[:, np.newaxis] >= x_left) & (a[:, np.newaxis] <= x_right)
    in_y = (b[:, np.newaxis] >= y_left) & (b[:, np.newaxis] <= y_right)
    
    inside_any = np.any(in_x & in_y, axis=1)
    
    fraction = np.mean(inside_any)
    V = fraction * (X1 - X0) * (Y1 - Y0)
    return V


def _generate_initial_positions():
    """Generate initial positions using a shear-aware spreading strategy."""
    N = 8
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    # Shear vectors
    shear_x = i_idx / N  # 0, 0, ..., 1, 1, ...
    shear_y = j_idx / N
    
    # Group tubes by their shear direction (i, j)
    # For each unique (i, j), we have exactly one tube
    # We want to place tubes with similar shear directions far apart
    
    # Strategy: use a grid in base space, but permute based on shear
    # Tubes with large shear (i=7, j=7) move a lot, so they should be placed
    # where they can "sweep" through space without overlapping others
    
    # Use a space-filling curve approach: sort by shear angle and distribute
    # evenly in a square region
    
    # Compute shear angle
    angles = np.arctan2(shear_y, shear_x)
    
    # Sort by angle to create a cyclic ordering
    order = np.argsort(angles)
    
    # Place in a grid pattern, but use the sorted order
    # Use an 8x8 grid in [-0.5, 0.5] x [-0.5, 0.5]
    grid_size = 0.8
    x_coords = np.linspace(-grid_size/2, grid_size/2, 8)
    y_coords = np.linspace(-grid_size/2, grid_size/2, 8)
    
    # Create grid positions
    gx, gy = np.meshgrid(x_coords, y_coords, indexing='ij')
    gx = gx.flatten()
    gy = gy.flatten()
    
    # Assign positions based on shear-angle ordering
    positions = np.zeros((64, 2))
    for sorted_rank, orig_idx in enumerate(order):
        # Place in a spiral or grid pattern
        row = sorted_rank // 8
        col = sorted_rank % 8
        positions[orig_idx, 0] = gx[row * 8 + col]
        positions[orig_idx, 1] = gy[row * 8 + col]
    
    return positions


def _local_search(positions, max_time=580, n_samples=30000):
    """Local search with coordinate descent and random perturbations."""
    start_time = time.time()
    current = positions.copy()
    current_V = _compute_volume_fast(current, n_samples=n_samples)
    best = current.copy()
    best_V = current_V
    
    N = 8
    rng = np.random.default_rng(42)
    
    # Multi-stage local search with decreasing step sizes
    step_sizes = [0.15, 0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for step in step_sizes:
        if time.time() - start_time > max_time - 30:
            break
        
        improved = True
        while improved and time.time() - start_time < max_time - 30:
            improved = False
            # Random order of tubes to avoid bias
            perm = rng.permutation(64)
            for k in perm:
                if time.time() - start_time > max_time - 30:
                    break
                
                for coord in range(2):
                    # Try multiple perturbations per coordinate
                    for _ in range(3):
                        delta = rng.uniform(-step, step)
                        trial = current.copy()
                        trial[k, coord] += delta
                        
                        # Keep positions in a reasonable range
                        if trial[k, coord] < -1.5 or trial[k, coord] > 1.5:
                            continue
                        
                        trial_V = _compute_volume_fast(trial, n_samples=n_samples)
                        if trial_V < current_V - 1e-12:
                            current = trial
                            current_V = trial_V
                            improved = True
                            if current_V < best_V:
                                best = current.copy()
                                best_V = current_V
    
    return best, best_V


def _refine_with_gradient(positions, max_time=60, n_samples=30000):
    """Final refinement using numerical gradient descent."""
    start_time = time.time()
    current = positions.copy()
    current_V = _compute_volume_fast(current, n_samples=n_samples)
    best = current.copy()
    best_V = current_V
    
    lr = 0.01
    decay = 0.95
    
    while time.time() - start_time < max_time:
        # Compute numerical gradient
        grad = np.zeros_like(current)
        eps = 1e-4
        for k in range(64):
            for coord in range(2):
                trial_plus = current.copy()
                trial_plus[k, coord] += eps
                V_plus = _compute_volume_fast(trial_plus, n_samples=n_samples)
                
                trial_minus = current.copy()
                trial_minus[k, coord] -= eps
                V_minus = _compute_volume_fast(trial_minus, n_samples=n_samples)
                
                grad[k, coord] = (V_plus - V_minus) / (2 * eps)
        
        # Gradient step
        current -= lr * grad
        # Keep in bounds
        current = np.clip(current, -1.5, 1.5)
        
        current_V = _compute_volume_fast(current, n_samples=n_samples)
        if current_V < best_V:
            best = current.copy()
            best_V = current_V
        
        lr *= decay
        if lr < 1e-6:
            break
    
    return best, best_V


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses shear-aware initialization, then refines with local search and gradient descent.
    """
    start_time = time.time()
    max_total_time = 620  # Leave 10s margin
    
    # Generate initial positions
    positions = _generate_initial_positions()
    
    # Local search phase
    if time.time() - start_time < max_total_time - 120:
        try:
            refined, _ = _local_search(positions, max_time=max_total_time - 100, n_samples=30000)
            positions = refined
        except Exception:
            pass
    
    # Gradient refinement phase
    if time.time() - start_time < max_total_time - 30:
        try:
            refined, _ = _refine_with_gradient(positions, max_time=max_total_time - 20, n_samples=30000)
            positions = refined
        except Exception:
            pass
    
    # Final deadline guard
    if time.time() - start_time > max_total_time:
        # Should not reach here, but just in case
        pass
    
    return positions
