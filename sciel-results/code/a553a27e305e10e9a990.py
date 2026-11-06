"""Bermudan exercise-policy research: backward induction with antithetic MC for continuation values."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on a state grid.
    Continuation values are estimated via antithetic Monte Carlo (50000 paths per state).
    Uses a denser grid focused on the put-relevant region.
    """
    start_time = time.time()
    DEADLINE = 142.0  # 8s safety margin before 150s kill

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    dt = T / n_exercise
    sqrt_dt = np.sqrt(dt)
    disc = np.exp(-r * dt)
    drift = (r - 0.5 * sigma * sigma) * dt

    # Ensure training_paths is a numpy array
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(-1, n_exercise)
    # Handle shape: (n_paths, n_exercise) or (n_exercise, n_paths)
    if paths.shape[0] != 4096 and paths.shape[1] == 4096:
        paths = paths.T
    if paths.shape[1] > n_exercise:
        paths = paths[:, :n_exercise]
    elif paths.shape[1] < n_exercise:
        pad_width = n_exercise - paths.shape[1]
        paths = np.hstack([paths, np.zeros((paths.shape[0], pad_width))])

    n_paths, n_dates = paths.shape

    # Build state grid: focus on put-relevant region with higher density at low states
    all_states = paths.flatten()
    # Use a log-spaced grid combined with quantiles for better coverage
    min_s = max(1.0, np.percentile(all_states, 0.5))
    max_s = np.percentile(all_states, 99.5)
    
    # Create a grid that is denser in the lower range where exercise decisions matter
    # For a put, the critical region is below K
    n_grid_low = 500   # points in [min_s, K]
    n_grid_high = 200  # points in [K, max_s]
    
    if K > min_s:
        grid_low = np.linspace(min_s, K, n_grid_low)
    else:
        grid_low = np.array([min_s])
    
    grid_high = np.linspace(K, max_s, n_grid_high)
    grid = np.concatenate([grid_low, grid_high[1:]])  # avoid duplicate at K
    grid = np.sort(grid)
    n_grid = len(grid)

    # Pre-generate antithetic normal random numbers for continuation estimation
    n_mc = 50000
    # Shape: (n_mc, 2) for antithetic pairs
    z_base = np.random.standard_normal(n_mc)
    z_antithetic = np.column_stack([z_base, -z_base])  # (n_mc, 2)
    n_total_mc = n_mc * 2
    z_flat = z_antithetic.flatten()  # (n_total_mc,)
    log_factor = drift + sigma * sqrt_dt * z_flat  # (n_total_mc,)

    # Backward induction
    # V[i] = optimal value at exercise date i (0-indexed) for state grid[j]
    V_grid = np.zeros((n_exercise, n_grid))
    cont_stored = np.zeros((n_exercise, n_grid))

    # Last exercise date: intrinsic value only (no continuation)
    V_grid[-1, :] = np.maximum(K - grid, 0.0)
    cont_stored[-1, :] = 0.0

    # Work backward
    for i in range(n_exercise - 2, -1, -1):
        # For each state grid[j], compute continuation value:
        # C(s) = disc * E[V[i+1](S_{i+1}) | S_i = s]
        # where S_{i+1} = s * exp(drift + sigma*sqrt_dt*Z)
        
        V_next_grid = V_grid[i + 1, :]
        
        cont_values = np.zeros(n_grid)
        chunk_size = 40
        for start in range(0, n_grid, chunk_size):
            end = min(start + chunk_size, n_grid)
            s_chunk = grid[start:end, None]  # (chunk, 1)
            S_next = s_chunk * np.exp(log_factor[None, :])  # (chunk, n_total_mc)
            
            # Interpolate V_next_grid at S_next
            S_next_clipped = np.clip(S_next, grid[0], grid[-1])
            V_next_vals = np.interp(S_next_clipped, grid, V_next_grid)  # (chunk, n_total_mc)
            
            # Average over MC paths
            cont_chunk = disc * np.mean(V_next_vals, axis=1)
            cont_values[start:end] = cont_chunk
        
        # Store continuation values
        cont_stored[i, :] = cont_values
        
        # Exercise value
        ex_values = np.maximum(K - grid, 0.0)
        
        # Optimal value
        V_grid[i, :] = np.maximum(ex_values, cont_values)
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            # Fill remaining with intrinsic value (conservative)
            for j in range(i, n_exercise):
                V_grid[j, :] = np.maximum(K - grid, 0.0)
                if j < n_exercise - 1:
                    cont_stored[j, :] = 0.0
            break

    # Create policy object
    class ExercisePolicy:
        def __init__(self, grid, cont_stored, n_exercise, K):
            self.grid = grid
            self.cont_stored = cont_stored
            self.n_exercise = n_exercise
            self.K = K
        
        def exercise(self, current_state, time_idx):
            """
            current_state: array of spot prices, shape (n_eval_paths,)
            time_idx: index of exercise date (0-indexed, 0 to n_exercise-1)
            Returns: boolean array, True if exercise
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            if time_idx < 0 or time_idx >= self.n_exercise:
                return np.zeros(len(current_state), dtype=bool)
            
            # Clip to grid range
            cs_clipped = np.clip(current_state, self.grid[0], self.grid[-1])
            
            # Interpolate continuation value at current state
            cont_vals = np.interp(cs_clipped, self.grid, self.cont_stored[time_idx, :])
            
            # Exercise if intrinsic >= continuation
            intrinsic = np.maximum(self.K - current_state, 0.0)
            
            return intrinsic >= cont_vals

    return ExercisePolicy(grid, cont_stored, n_exercise, K)
# EVOLVE-BLOCK-END
