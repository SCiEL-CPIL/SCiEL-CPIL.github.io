# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_volume_fast(positions, seed=42, n_samples=20000):
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

def _compute_volume(positions, seed=12345, n_samples=500000):
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
            step *= 0.85
            if step < 0.0005:
                break
    return best, best_V

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    max_total_time = 580
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    # Multiple initializations
    inits = []
    # Centered
    inits.append(np.stack([-i_flat / 16.0, -j_flat / 16.0], axis=1))
    # Tighter
    inits.append(np.stack([-i_flat / 12.0, -j_flat / 12.0], axis=1))
    inits.append(np.stack([-i_flat / 20.0, -j_flat / 20.0], axis=1))
    # Random perturbation of centered
    rng_init = np.random.default_rng(7)
    base = np.stack([-i_flat / 16.0, -j_flat / 16.0], axis=1)
    for _ in range(3):
        pert = rng_init.normal(0, 0.05, size=(64, 2))
        inits.append(base + pert)
    
    best_positions = inits[0].copy()
    best_V = _compute_volume_fast(inits[0], n_samples=20000, seed=123)
    
    for init in inits[1:]:
        if time.time() - start_time > max_total_time - 60:
            break
        V = _compute_volume_fast(init, n_samples=20000, seed=123)
        if V < best_V:
            best_positions = init.copy()
            best_V = V
    
    # Local search on best
    if time.time() - start_time < max_total_time - 30:
        refined, V_ref = _local_search(best_positions, max_time=min(100, max_total_time - 40), n_samples=10000, seed=1)
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Second round: try more random restarts
    rng2 = np.random.default_rng(55)
    for _ in range(5):
        if time.time() - start_time > max_total_time - 30:
            break
        pert = rng2.normal(0, 0.03, size=(64, 2))
        trial = best_positions + pert
        V_trial = _compute_volume_fast(trial, n_samples=10000, seed=789)
        if V_trial < best_V:
            best_positions = trial.copy()
            best_V = V_trial
            if time.time() - start_time < max_total_time - 30:
                refined, V_ref = _local_search(best_positions, max_time=20.0, n_samples=10000, seed=999)
                if V_ref < best_V:
                    best_positions = refined
                    best_V = V_ref
    
    return best_positions

# EVOLVE-BLOCK-END
