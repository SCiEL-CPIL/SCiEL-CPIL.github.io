import numpy as np
import time

def _compute_volume(positions, N=8, n_samples=500000, seed=12345):
    """
    Compute the Monte Carlo volume estimate for given positions.
    positions: (64, 2) array.
    """
    rng = np.random.default_rng(seed)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    x_starts = positions[:, 0]
    x_ends = positions[:, 0] + (i_flat + 1) / N
    y_starts = positions[:, 1]
    y_ends = positions[:, 1] + (j_flat + 1) / N
    
    X0 = np.min(x_starts)
    X1 = np.max(x_ends)
    Y0 = np.min(y_starts)
    Y1 = np.max(y_ends)
    
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return float('inf')
    
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    inside = np.zeros(n_samples, dtype=bool)
    
    # Vectorized check for all tubes
    # x_k + c * i_k / N <= a <= x_k + c * i_k / N + 1/N
    # y_k + c * j_k / N <= b <= y_k + c * j_k / N + 1/N
    
    # We can loop over tubes to keep memory low and code simple, 
    # or vectorize over tubes.
    # 64 tubes is small. Looping is fine.
    for k in range(64):
        i_k = i_flat[k]
        j_k = j_flat[k]
        
        x_lo = positions[k, 0] + c * i_k / N
        x_hi = x_lo + 1.0 / N
        y_lo = positions[k, 1] + c * j_k / N
        y_hi = y_lo + 1.0 / N
        
        inside |= (a >= x_lo) & (a <= x_hi) & (b >= y_lo) & (b <= y_hi)
        
    frac = np.mean(inside)
    return frac * (X1 - X0) * (Y1 - Y0)

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # Initial guess: centered shear
    base_x = -np.arange(N) / (2 * N)
    base_y = -np.arange(N) / (2 * N)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x_flat = base_x[i_grid.flatten()]
    base_y_flat = base_y[j_grid.flatten()]
    
    pos = np.stack([base_x_flat, base_y_flat], axis=1)
    
    # Optimization
    start_time = time.time()
    deadline = start_time + 590  # 10s margin
    
    # Get initial volume with full precision
    best_V = _compute_volume(pos, n_samples=500000)
    best_pos = pos.copy()
    current_pos = pos.copy()
    
    # Parameters for search
    step_size = 0.05
    min_step_size = 0.001
    noise_samples = 50000  # Lower precision for speed
    
    # We can try to reduce dimensionality by fixing translation.
    # But let's just run random search.
    
    while time.time() < deadline:
        # Generate candidate
        # Use Gaussian noise
        noise = np.random.normal(0, step_size, size=(64, 2))
        candidate = current_pos + noise
        
        # Evaluate with lower precision
        V_cand = _compute_volume(candidate, n_samples=noise_samples)
        
        # Accept if better
        # To avoid noise, we can require a small improvement or just accept and see
        # If we accept noisy improvements, we might drift.
        # Let's use a "best so far" tracker with high precision verification.
        
        if V_cand < best_V:
            # Potential improvement, verify with high precision
            V_verify = _compute_volume(candidate, n_samples=500000)
            if V_verify < best_V:
                best_V = V_verify
                best_pos = candidate.copy()
                current_pos = candidate.copy()
                # Increase step size slightly or keep same
                step_size = max(min_step_size, step_size * 0.95)
            else:
                # False positive, reject
                step_size = max(min_step_size, step_size * 0.9)
        else:
            # Reject, shrink step size
            step_size = max(min_step_size, step_size * 0.95)
            
        # Adaptive step size: if we haven't improved in a while, shrink more?
        # Or just let it decay.
        
    return best_pos
