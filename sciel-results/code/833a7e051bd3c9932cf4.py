# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_volume_fast(positions, seed=42, n_samples=20000):
    """
    Fast Monte Carlo volume estimate.
    """
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
    
    # Vectorized check for all tubes? 
    # a, b, c are (n_samples,)
    # x, y are (64,)
    # We can broadcast.
    # x_k + c * i_k / N
    # Shape of x_left: (64, n_samples) if we do x[:, None] + c[None, :] * i_idx[:, None] / N
    # This creates a 64 x 20000 array, which is 1.2M floats. Manageable.
    
    x_left = x[:, None] + c[None, :] * i_idx[:, None] / N
    x_right = x_left + 1.0 / N
    y_left = y[:, None] + c[None, :] * j_idx[:, None] / N
    y_right = y_left + 1.0 / N
    
    # in_x[k, m] is True if a[m] is in [x_left[k, m], x_right[k, m]]
    in_x = (a[None, :] >= x_left) & (a[None, :] <= x_right)
    in_y = (b[None, :] >= y_left) & (b[None, :] <= y_right)
    
    # inside[m] is True if any tube k contains point m
    inside = np.any(in_x & in_y, axis=0)
    
    fraction = np.mean(inside)
    V = fraction * (X1 - X0) * (Y1 - Y0)
    return V

def _compute_volume(positions, seed=12345, n_samples=500000):
    """
    Exact Monte Carlo volume estimate as per task description.
    """
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
    
    x_left = x[:, None] + c[None, :] * i_idx[:, None] / N
    x_right = x_left + 1.0 / N
    y_left = y[:, None] + c[None, :] * j_idx[:, None] / N
    y_right = y_left + 1.0 / N
    
    in_x = (a[None, :] >= x_left) & (a[None, :] <= x_right)
    in_y = (b[None, :] >= y_left) & (b[None, :] <= y_right)
    
    inside = np.any(in_x & in_y, axis=0)
    
    fraction = np.mean(inside)
    V = fraction * (X1 - X0) * (Y1 - Y0)
    return V

def _local_search(positions, max_time=50.0, n_samples=10000, seed=42):
    """
    Coordinate descent local search.
    """
    start_time = time.time()
    current = positions.copy()
    current_V = _compute_volume_fast(current, n_samples=n_samples, seed=seed)
    best = current.copy()
    best_V = current_V
    
    rng = np.random.default_rng(seed)
    step = 0.05
    
    while time.time() - start_time < max_time:
        improved = False
        indices = rng.permutation(64)
        
        for k in indices:
            if time.time() - start_time > max_time:
                break
                
            for coord in [0, 1]:
                for direction in [1, -1]:
                    trial = current.copy()
                    trial[k, coord] += direction * step
                    trial_V = _compute_volume_fast(trial, n_samples=n_samples, seed=seed)
                    
                    if trial_V < current_V - 1e-12:
                        current = trial
                        current_V = trial_V
                        improved = True
                        if current_V < best_V:
                            best = current.copy()
                            best_V = current_V
                        break
        if not improved:
            step *= 0.9
            if step < 0.001:
                break
    return best, best_V

def kakeya_needle_3d8() -> np.ndarray:
    """
    Generate the 64 base positions.
    """
    N = 8
    start_time = time.time()
    max_total_time = 600
    
    # Initialization: Centered configuration
    # x_k = -i_k / 16, y_k = -j_k / 16
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    init_x = -i_flat / 16.0
    init_y = -j_flat / 16.0
    initial_positions = np.stack([init_x, init_y], axis=1)
    
    # Evaluate initial
    V_init = _compute_volume_fast(initial_positions, n_samples=20000, seed=123)
    
    best_positions = initial_positions.copy()
    best_V = V_init
    
    # Local search refinement
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 60:
        search_time = min(120.0, time_remaining - 30)
        refined, V_ref = _local_search(best_positions, max_time=search_time, n_samples=10000, seed=1)
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Try a few variations of the centered configuration (scaling)
    # x_k = s * (-i_k/16) + o
    # The centered one has s=1, o=0.
    # Maybe s=1.1 or s=0.9 helps?
    rng = np.random.default_rng(99)
    for _ in range(5):
        if time.time() - start_time > max_total_time - 30:
            break
        s = rng.uniform(0.8, 1.2)
        o = rng.uniform(-0.1, 0.1)
        trial_pos = np.stack([s * init_x + o, s * init_y + o], axis=1)
        V_trial = _compute_volume_fast(trial_pos, n_samples=10000, seed=456)
        if V_trial < best_V:
            best_positions = trial_pos
            best_V = V_trial
            # Refine this new best
            if time.time() - start_time < max_total_time - 30:
                refined, V_ref = _local_search(best_positions, max_time=20.0, n_samples=10000, seed=789)
                if V_ref < best_V:
                    best_positions = refined
                    best_V = V_ref

    return best_positions

# EVOLVE-BLOCK-END
