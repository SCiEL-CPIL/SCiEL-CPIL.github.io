# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_hermite

def vectorized_interp(grid, values, queries):
    queries = np.asarray(queries)
    original_shape = queries.shape
    queries_flat = queries.flatten()
    
    idx = np.searchsorted(grid, queries_flat)
    
    lower_mask = idx == 0
    upper_mask = idx == len(grid)
    valid_mask = ~(lower_mask | upper_mask)
    
    result = np.empty_like(queries_flat)
    
    result[lower_mask] = values[0]
    result[upper_mask] = values[-1]
    
    if np.any(valid_mask):
        idx_v = idx[valid_mask]
        q_v = queries_flat[valid_mask]
        
        x0 = grid[idx_v - 1]
        x1 = grid[idx_v]
        y0 = values[idx_v - 1]
        y1 = values[idx_v]
        
        slope = (y1 - y0) / (x1 - x0)
        result[valid_mask] = y0 + slope * (q_v - x0)
        
    return result.reshape(original_shape)

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    
    dt = T / n_steps
    df = np.exp(-r * dt)
    sqrt_dt = np.sqrt(dt)
    
    log_s_min = np.log(1.0)
    log_s_max = np.log(500.0)
    n_grid = 2000
    log_grid = np.linspace(log_s_min, log_s_max, n_grid)
    grid = np.exp(log_grid)
    
    V = np.zeros((n_steps, n_grid))
    V[-1, :] = np.maximum(K - grid, 0.0)
    
    n_quad = 64
    gh_nodes, gh_weights = roots_hermite(n_quad)
    z_nodes = np.sqrt(2.0) * gh_nodes
    z_weights = gh_weights / np.sqrt(np.pi)
    
    drift = (r - sigma**2 / 2.0) * dt
    diffusion = sigma * sqrt_dt
    
    for i in range(n_steps - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Vectorized computation
        # queries shape: (n_grid, n_quad)
        log_queries = log_grid[:, None] + drift + diffusion * z_nodes[None, :]
        queries = np.exp(log_queries)
        
        # Interpolate V[i+1] at queries
        v_next = vectorized_interp(grid, V[i + 1, :], queries)
        
        # Continuation value: df * sum(w * v)
        continuation = df * np.sum(z_weights * v_next, axis=1)
        
        exercise_val = np.maximum(K - grid, 0.0)
        V[i, :] = np.maximum(exercise_val, continuation)
        
    class ExercisePolicy:
        def __init__(self, K, r, sigma, T, n_steps, grid, log_grid, V, z_nodes, z_weights, drift, diffusion, df):
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_steps = n_steps
            self.grid = grid
            self.log_grid = log_grid
            self.V = V
            self.z_nodes = z_nodes
            self.z_weights = z_weights
            self.drift = drift
            self.diffusion = diffusion
            self.df = df
            
        def exercise(self, current_state, time_idx):
            if isinstance(time_idx, (int, np.integer)):
                idx = int(time_idx)
            else:
                dt_local = self.T / self.n_steps
                idx = int(time_idx / dt_local)
                idx = min(idx, self.n_steps - 1)
                idx = max(idx, 0)
            
            s_arr = np.array(current_state, dtype=float)
            
            if idx == self.n_steps - 1:
                return s_arr < self.K
            
            exercise_val = np.maximum(self.K - s_arr, 0.0)
            
            # Vectorized continuation
            # s_arr shape: (M,)
            # queries shape: (M, n_quad)
            log_queries = np.log(s_arr)[:, None] + self.drift + self.diffusion * self.z_nodes[None, :]
            queries = np.exp(log_queries)
            
            v_next = vectorized_interp(self.grid, self.V[idx + 1, :], queries)
            continuation = self.df * np.sum(self.z_weights * v_next, axis=1)
            
            return exercise_val >= continuation
            
    return ExercisePolicy(K, r, sigma, T, n_steps, grid, log_grid, V, z_nodes, z_weights, drift, diffusion, df)
# EVOLVE-BLOCK-END
