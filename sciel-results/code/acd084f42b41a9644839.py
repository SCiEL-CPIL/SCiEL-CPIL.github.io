# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_volume(positions, seed=12345, n_samples=500000):
    """Compute the union volume using the exact Monte Carlo method from the task."""
    N = 8
    x = positions[:, 0]
    y = positions[:, 1]
    
    # Get i and j indices for each tube
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    # Compute box bounds
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
    
    # Check if each point is inside at least one tube
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


def _compute_volume_fast(positions, seed=42, n_samples=20000):
    """Faster volume estimate for local search iterations."""
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


def _local_search(positions, max_time=60.0, n_samples=15000, seed=42, start_step=0.08):
    """
    Perform local search using coordinate descent with adaptive step sizes.
    Returns the best positions found within the time limit.
    """
    start_time = time.time()
    current = positions.copy()
    current_V = _compute_volume_fast(current, n_samples=n_samples, seed=seed)
    best = current.copy()
    best_V = current_V
    
    N = 8
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    rng = np.random.default_rng(seed)
    
    step = start_step
    no_improve_count = 0
    max_no_improve = 8
    
    while time.time() - start_time < max_time:
        improved = False
        indices = rng.permutation(64)
        
        for k in indices:
            if time.time() - start_time > max_time:
                break
                
            for coord in [0, 1]:
                # Try multiple step sizes for each direction
                for direction in [1, -1]:
                    for step_mult in [1.0, 0.5, 0.25]:
                        actual_step = step * step_mult
                        trial = current.copy()
                        trial[k, coord] += direction * actual_step
                        
                        # Keep positions in reasonable range
                        if trial[k, coord] < -1.0 or trial[k, coord] > 2.5:
                            continue
                        
                        trial_V = _compute_volume_fast(trial, n_samples=n_samples, seed=seed)
                        
                        if trial_V < current_V - 1e-10:
                            current = trial
                            current_V = trial_V
                            improved = True
                            
                            if current_V < best_V:
                                best = current.copy()
                                best_V = current_V
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                            
                            step *= 0.92
                            break
                    else:
                        continue
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Reset step size and add perturbation
                step = max(0.03, step * 1.5)
                no_improve_count = 0
                perturbation = rng.normal(0, 0.03, size=current.shape)
                current += perturbation
                current_V = _compute_volume_fast(current, n_samples=n_samples, seed=seed)
                if current_V < best_V:
                    best = current.copy()
                    best_V = current_V
            else:
                step *= 0.85
                if step < 0.005:
                    step = 0.02
    
    return best, best_V


def _simulated_annealing(positions, max_time=120.0, n_samples=15000, seed=77):
    """
    Simulated annealing for escaping local minima.
    """
    start_time = time.time()
    current = positions.copy()
    current_V = _compute_volume_fast(current, n_samples=n_samples, seed=seed)
    best = current.copy()
    best_V = current_V
    
    rng = np.random.default_rng(seed)
    
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling = 0.995
    
    while time.time() - start_time < max_time and T > T_min:
        trial = current.copy()
        # Perturb 2-4 random coordinates
        n_perturb = rng.integers(2, 5)
        indices = rng.choice(64, size=n_perturb, replace=False)
        for k in indices:
            trial[k, 0] += rng.normal(0, 0.05)
            trial[k, 1] += rng.normal(0, 0.05)
        
        # Clamp to reasonable range
        trial[:, 0] = np.clip(trial[:, 0], -1.0, 2.5)
        trial[:, 1] = np.clip(trial[:, 1], -1.0, 2.5)
        
        trial_V = _compute_volume_fast(trial, n_samples=n_samples, seed=seed)
        
        # Accept if better or with probability exp(-(delta)/T)
        if trial_V < current_V:
            current = trial
            current_V = trial_V
            if current_V < best_V:
                best = current.copy()
                best_V = current_V
        else:
            delta = trial_V - current_V
            if delta < 0.01 and np.log(rng.uniform()) > -delta / T:
                current = trial
                current_V = trial_V
        
        T *= cooling
    
    return best, best_V


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses Keich binary construction as initialization, then refines with
    multi-phase local search and simulated annealing to minimize union volume.
    """
    N = 8
    
    # Keich binary construction for n = 2^k base positions
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
    initial_positions = np.stack([base_x, base_y], axis=1)
    
    start_time = time.time()
    max_total_time = 600  # Safety margin before 630s limit
    
    best_positions = initial_positions.copy()
    best_V = _compute_volume(initial_positions, n_samples=50000)
    
    # Phase 1: Quick local search on the initial guess
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 40:
        refined, V_ref = _local_search(initial_positions, max_time=30.0, n_samples=10000, seed=1, start_step=0.1)
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Phase 2: Simulated annealing to escape local minima
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 80:
        refined, V_ref = _simulated_annealing(best_positions, max_time=60.0, n_samples=10000, seed=55)
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Phase 3: Multi-start local search from perturbations
    rng = np.random.default_rng(123)
    num_starts = 4
    time_per_start = (max_total_time - (time.time() - start_time) - 60) / num_starts
    if time_per_start < 15:
        num_starts = max(1, int((max_total_time - (time.time() - start_time) - 60) / 15))
        time_per_start = (max_total_time - (time.time() - start_time) - 60) / num_starts if num_starts > 0 else 0
    
    for i in range(num_starts):
        if time.time() - start_time > max_total_time - 60:
            break
            
        noise = rng.normal(0, 0.04, size=best_positions.shape)
        start_pos = best_positions + noise
        start_pos[:, 0] = np.clip(start_pos[:, 0], -0.5, 2.0)
        start_pos[:, 1] = np.clip(start_pos[:, 1], -0.5, 2.0)
        
        refined, V_ref = _local_search(start_pos, max_time=time_per_start, n_samples=15000, seed=200+i, start_step=0.06)
        
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Phase 4: Final refinement with higher precision
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 20:
        final_time = min(40.0, time_remaining - 5)
        if final_time > 10:
            refined, V_ref = _local_search(best_positions, max_time=final_time, n_samples=25000, seed=999, start_step=0.03)
            if V_ref < best_V:
                best_positions = refined
                best_V = V_ref
    
    return best_positions


# EVOLVE-BLOCK-END
