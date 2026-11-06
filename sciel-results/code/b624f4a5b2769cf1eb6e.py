"""Bermudan exercise-policy research: backward induction with antithetic MC for continuation values."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on a state grid.
    Continuation values are estimated via antithetic Monte Carlo (20000 paths per state).
    """
    start_time = time.time()
    DEADLINE = 135.0  # 15s safety margin before 150s kill

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

    # Build state grid using quantiles of all observed states
    all_states = paths.flatten()
    n_grid = 400
    grid = np.quantile(all_states, np.linspace(0, 1, n_grid))
    grid = np.unique(grid)
    grid = np.sort(grid)
    n_grid = len(grid)

    # Pre-generate antithetic normal random numbers for continuation estimation
    n_mc = 20000
    # Shape: (n_mc, 2) for antithetic pairs
    z_base = np.random.standard_normal(n_mc)
    z_antithetic = np.column_stack([z_base, -z_base])  # (n_mc, 2)
    n_total_mc = n_mc * 2

    # Backward induction
    # V[i] = optimal value at exercise date i (0-indexed) for state grid[j]
    V_grid = np.zeros((n_exercise, n_grid))

    # Last exercise date: intrinsic value only (no continuation)
    V_grid[-1, :] = np.maximum(K - grid, 0.0)

    # Work backward
    for i in range(n_exercise - 2, -1, -1):
        # For each state grid[j], compute continuation value:
        # C(s) = disc * E[V[i+1](S_{i+1}) | S_i = s]
        # where S_{i+1} = s * exp(drift + sigma*sqrt_dt*Z)
        
        # Vectorized: for all grid points at once
        # S_next = grid[:, None] * exp(drift + sigma*sqrt_dt * z[None, :])
        # This is (n_grid, n_total_mc)
        
        # To save memory, process in chunks if needed
        chunk_size = 50
        V_next_grid = V_grid[i + 1, :]
        
        cont_values = np.zeros(n_grid)
        for start in range(0, n_grid, chunk_size):
            end = min(start + chunk_size, n_grid)
            s_chunk = grid[start:end, None]  # (chunk, 1)
            # S_next = s_chunk * exp(drift + sigma*sqrt_dt * z)
            log_factor = drift + sigma * sqrt_dt * z_antithetic.T  # (n_total_mc, 2) -> need (n_total_mc,)
            # Actually z_antithetic is (n_mc, 2), so z_antithetic.T is (2, n_mc)
            # Let's flatten: z_flat = z_antithetic.flatten() -> (n_total_mc,)
            z_flat = z_antithetic.flatten()  # (n_total_mc,)
            log_factor = drift + sigma * sqrt_dt * z_flat  # (n_total_mc,)
            S_next = s_chunk * np.exp(log_factor[None, :])  # (chunk, n_total_mc)
            
            # Interpolate V_next_grid at S_next
            # Clip S_next to grid range for interpolation
            S_next_clipped = np.clip(S_next, grid[0], grid[-1])
            V_next_vals = np.interp(S_next_clipped, grid, V_next_grid)  # (chunk, n_total_mc)
            
            # Average over MC paths
            cont_chunk = disc * np.mean(V_next_vals, axis=1)
            cont_values[start:end] = cont_chunk
        
        # Exercise value
        ex_values = np.maximum(K - grid, 0.0)
        
        # Optimal value
        V_grid[i, :] = np.maximum(ex_values, cont_values)
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break

    # Build exercise policy: exercise at date i if (K - S) >= continuation value
    # We need to store the continuation values to compare at exercise time
    # Actually, we can recompute: exercise if V[i] == ex_values (i.e., ex >= cont)
    # But floating point: exercise if (K - S) >= cont_value
    
    # Store continuation values for each date and grid point
    # Recompute cont values for the policy
    cont_stored = np.zeros((n_exercise, n_grid))
    for i in range(n_exercise - 1):  # Last date always exercises if in the money
        ex_values = np.maximum(K - grid, 0.0)
        # Recompute continuation (or store during backward induction)
        # For efficiency, let's recompute
        V_next_grid = V_grid[i + 1, :]
        chunk_size = 50
        for start in range(0, n_grid, chunk_size):
            end = min(start + chunk_size, n_grid)
            s_chunk = grid[start:end, None]
            z_flat = z_antithetic.flatten()
            log_factor = drift + sigma * sqrt_dt * z_flat
            S_next = s_chunk * np.exp(log_factor[None, :])
            S_next_clipped = np.clip(S_next, grid[0], grid[-1])
            V_next_vals = np.interp(S_next_clipped, grid, V_next_grid)
            cont_chunk = disc * np.mean(V_next_vals, axis=1)
            cont_stored[i, start:end] = cont_chunk
    
    # For the last date, continuation is 0 (or we just exercise if in the money)
    cont_stored[-1, :] = 0.0

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
