# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_volume(positions, seed=12345, n_samples=50000):
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


def _local_search(positions, max_time=60.0, n_samples=20000, seed=42):
    """
    Perform local search using coordinate descent with random perturbations.
    Returns the best positions found within the time limit.
    """
    start_time = time.time()
    current = positions.copy()
    current_V = _compute_volume(current, n_samples=n_samples)
    best = current.copy()
    best_V = current_V
    
    N = 8
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    rng = np.random.default_rng(seed)
    
    # Adaptive step sizes
    step = 0.1
    
    # We will iterate until time runs out
    # Each iteration tries to move one coordinate in one direction
    # We use a simple hill climbing with random restarts of step size
    
    while time.time() - start_time < max_time:
        improved = False
        # Try each tube and coordinate
        # Randomize the order to avoid bias
        indices = rng.permutation(64)
        
        for k in indices:
            if time.time() - start_time > max_time:
                break
                
            for coord in [0, 1]:
                # Try positive and negative steps
                for direction in [1, -1]:
                    trial = current.copy()
                    # Add some randomness to the step size to escape plateaus
                    actual_step = step * rng.uniform(0.5, 1.5)
                    trial[k, coord] += direction * actual_step
                    
                    # Bounds check: keep positions in a reasonable range
                    # The box size is roughly 1.0 to 1.5. Positions should stay near [0, 1.5]
                    if trial[k, coord] < -0.5 or trial[k, coord] > 2.0:
                        continue
                    
                    trial_V = _compute_volume(trial, n_samples=n_samples)
                    
                    if trial_V < current_V - 1e-12:
                        current = trial
                        current_V = trial_V
                        improved = True
                        
                        if current_V < best_V:
                            best = current.copy()
                            best_V = current_V
                            
                        # Reduce step size slightly on improvement
                        step *= 0.95
                        break # Move to next coordinate/tube
        
        if not improved:
            # If no improvement, reduce step size or reset to a slightly larger step
            # to try to jump over local minima
            step *= 0.8
            if step < 0.001:
                # Reset step size and add a random perturbation to current to restart
                step = 0.05
                perturbation = rng.normal(0, 0.02, size=current.shape)
                current += perturbation
                current_V = _compute_volume(current, n_samples=n_samples)
                if current_V < best_V:
                    best = current.copy()
                    best_V = current_V

    return best, best_V


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses Keich binary construction as initialization, then refines with a 
    multi-start local search strategy to maximize the score (minimize volume).
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
    max_total_time = 620  # Leave 10s margin for the evaluator's 630s limit
    
    best_positions = initial_positions.copy()
    best_V = _compute_volume(initial_positions, n_samples=50000)
    
    # Phase 1: Quick local search on the initial guess
    # Use fewer samples for speed
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 30:
        refined, V_ref = _local_search(initial_positions, max_time=30.0, n_samples=10000, seed=1)
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Phase 2: Multi-start local search from perturbations of the best found so far
    # We will run several shorter searches with different random seeds/perturbations
    rng = np.random.default_rng(123)
    num_starts = 5
    time_per_start = (max_total_time - (time.time() - start_time) - 60) / num_starts
    # Ensure time_per_start is at least 10s, otherwise reduce num_starts
    if time_per_start < 10:
        num_starts = max(1, int((max_total_time - (time.time() - start_time) - 60) / 10))
        time_per_start = (max_total_time - (time.time() - start_time) - 60) / num_starts if num_starts > 0 else 0
        
    for i in range(num_starts):
        if time.time() - start_time > max_total_time - 60:
            break
            
        # Create a perturbed starting point
        # Small perturbation around the current best
        noise = rng.normal(0, 0.05, size=best_positions.shape)
        start_pos = best_positions + noise
        
        # Run local search
        refined, V_ref = _local_search(start_pos, max_time=time_per_start, n_samples=20000, seed=100+i)
        
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref

    # Phase 3: Final refinement with higher precision samples
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 10:
        # Use a slightly larger sample size for the final polish
        # but limit time to ensure we return
        final_time = min(60.0, time_remaining - 5)
        if final_time > 5:
            refined, V_ref = _local_search(best_positions, max_time=final_time, n_samples=50000, seed=999)
            if V_ref < best_V:
                best_positions = refined
                best_V = V_ref

    return best_positions


# EVOLVE-BLOCK-END
