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
    # Tube k at height c: [x_k + c*i_k/N, x_k + c*i_k/N + 1/N] x [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
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


def _local_search(positions, max_time=580, n_samples=50000):
    """Local search with adaptive step sizing, multiple restarts, and final polish."""
    start_time = time.time()
    best = positions.copy()
    best_V = _compute_volume(best, n_samples=n_samples)
    
    N = 8
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    rng = np.random.default_rng(42)
    
    def _coordinate_descent(current, current_V, max_steps, step_sizes, time_budget):
        """Run coordinate descent with given parameters."""
        improved = True
        steps_done = 0
        for step in step_sizes:
            if time.time() - start_time > max_time - 30:
                break
            improved = True
            while improved and steps_done < max_steps and time.time() - start_time < max_time - 30:
                improved = False
                for k in range(64):
                    if time.time() - start_time > max_time - 30:
                        break
                    for coord in range(2):
                        for direction in [-1, 1]:
                            trial = current.copy()
                            trial[k, coord] += direction * step * rng.uniform(0.5, 1.5)
                            
                            # Keep positions reasonable
                            if trial[k, coord] < -1 or trial[k, coord] > 2:
                                continue
                            
                            trial_V = _compute_volume(trial, n_samples=n_samples)
                            if trial_V < current_V - 1e-10:
                                current = trial
                                current_V = trial_V
                                improved = True
                                steps_done += 1
                                if current_V < best_V:
                                    best = current.copy()
                                    best_V = current_V
        return current, current_V
    
    # Phase 1: Main coordinate descent from initial position
    step_sizes_main = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    current, current_V = _coordinate_descent(positions.copy(), _compute_volume(positions, n_samples=n_samples), 
                                              max_steps=200, step_sizes=step_sizes_main, time_budget=max_time - 60)
    
    # Phase 2: Multiple restarts with perturbations
    n_restarts = 3
    for r in range(n_restarts):
        if time.time() - start_time > max_time - 30:
            break
        
        # Create perturbed initialization
        perturbation_scale = 0.05 * (r + 1)
        perturbed = best + rng.uniform(-perturbation_scale, perturbation_scale, size=best.shape)
        # Clamp to reasonable bounds
        perturbed = np.clip(perturbed, -1, 2)
        
        perturbed_V = _compute_volume(perturbed, n_samples=n_samples)
        if perturbed_V < best_V:
            best = perturbed.copy()
            best_V = perturbed_V
        
        # Run coordinate descent from perturbed position
        step_sizes_restart = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        current, current_V = _coordinate_descent(perturbed, perturbed_V, 
                                                  max_steps=100, step_sizes=step_sizes_restart, time_budget=max_time - 30)
    
    # Phase 3: Final high-precision polish
    if time.time() - start_time < max_time - 15:
        step_sizes_polish = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]
        current, current_V = _coordinate_descent(best.copy(), best_V, 
                                                  max_steps=50, step_sizes=step_sizes_polish, time_budget=max_time - 10)
    
    return best, best_V


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses Keich binary construction as initialization, then refines with local search.
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
    positions = np.stack([base_x, base_y], axis=1)
    
    # Deadline guard
    start_time = time.time()
    max_total_time = 620  # Leave 10s margin
    
    # Try local search if we have time
    if time.time() - start_time < max_total_time - 100:
        try:
            refined, _ = _local_search(positions, max_time=max_total_time - 20)
            positions = refined
        except Exception:
            pass  # Keep original if search fails
    
    return positions


# EVOLVE-BLOCK-END
