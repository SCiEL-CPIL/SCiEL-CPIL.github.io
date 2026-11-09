"""Bermudan put exercise policy via high-resolution backward induction on a log-grid with optimized interpolation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)

    dt = T / n_exercise
    disc = np.exp(-r * dt)

    # Grid setup: log-spaced grid
    # Increased resolution for better boundary capture and accuracy
    n_grid = 16384
    log_S_min = np.log(0.01 * K)
    log_S_max = np.log(10.0 * K)
    log_grid = np.linspace(log_S_min, log_S_max, n_grid)
    S_grid = np.exp(log_grid)
    step = (log_S_max - log_S_min) / (n_grid - 1)
    
    # Precompute intrinsic values on the grid
    intrinsic_grid = np.maximum(K - S_grid, 0.0)

    # Gauss-Hermite quadrature with higher order for accuracy
    from numpy.polynomial.hermite import hermgauss
    n_quad = 128
    gh_nodes, gh_weights = hermgauss(n_quad)

    # mu and vol for the log-return
    mu = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    # Quadrature points for log(S_next/s)
    # log_ratio = mu + vol * sqrt(2) * gh_nodes
    log_ratio = mu + vol * np.sqrt(2.0) * gh_nodes
    # Weights scaled by 1/sqrt(pi) for standard normal
    quad_weights = gh_weights / np.sqrt(np.pi)
    
    # Precompute exp(log_ratio) for potential use, though we use log-space interpolation
    # exp_log_ratio = np.exp(log_ratio)

    def fast_interp_uniform(x, x_min, dx, y, K_val):
        """
        Fast linear interpolation on a uniform grid in log-space.
        x: array of log-prices
        x_min, dx: grid start and step
        y: values on the grid
        K_val: strike price for boundary handling
        """
        n_y = len(y)
        # Calculate floating point indices
        idx_float = (x - x_min) / dx
        idx_floor = np.floor(idx_float).astype(np.int64)
        
        # Handle boundaries
        # Left boundary: S < S_min (x < x_min)
        # For a put, if S is very low, V(S) approx K - S
        # We need S = exp(x)
        left_mask = idx_floor < 0
        right_mask = idx_float >= (n_y - 1)
        
        # Clip indices for interpolation
        idx_floor_clipped = np.clip(idx_floor, 0, n_y - 2)
        idx_ceil_clipped = idx_floor_clipped + 1
        
        frac = idx_float - idx_floor
        # Ensure frac is within [0, 1) for clipped indices
        # If idx_floor was clipped, frac needs adjustment? 
        # Actually, if we clip idx_floor, the corresponding x is outside the grid.
        # We handle outside grid with masks.
        
        y_interp = y[idx_floor_clipped] * (1 - frac) + y[idx_ceil_clipped] * frac
        
        # Apply left boundary: V(S) = K - S
        if np.any(left_mask):
            S_left = np.exp(x[left_mask])
            y_interp[left_mask] = np.maximum(K_val - S_left, 0.0)
            
        # Apply right boundary: V(S) = 0
        if np.any(right_mask):
            y_interp[right_mask] = 0.0
            
        return y_interp

    # Backward induction
    # V[i] is the value at exercise date i (0-indexed, 0 = first exercise date)
    # V[n_exercise-1] = max(K - S, 0)
    
    V = np.empty((n_exercise, n_grid))
    
    # Last exercise date
    V[-1] = intrinsic_grid

    # Store continuation values for policy determination
    # We need cont_values[i] to decide exercise at step i
    # Exercise if intrinsic >= continuation
    # We can store the continuation values directly
    cont_values = [None] * n_exercise

    for i in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 5.0:
            # Fill remaining with intrinsic value as fallback
            for j in range(i, -1, -1):
                V[j] = intrinsic_grid
                if j < n_exercise - 1:
                     cont_values[j] = intrinsic_grid # Fallback: exercise if ITM
            break
        
        # Compute continuation value from V[i+1]
        # V_next = V[i+1]
        # For each grid point log_grid[k], the quadrature points are log_grid[k] + log_ratio[j]
        # We need to interpolate V_next at these points.
        
        # Vectorized computation using fast_interp_uniform
        # We process in chunks to manage memory if needed, but 16384 x 128 is ~2M floats, ~16MB.
        # It's safe to do it in one go or small chunks.
        
        # Create the matrix of quadrature log positions
        # quad_pos[k, j] = log_grid[k] + log_ratio[j]
        # This is (16384, 128).
        
        # To avoid creating a huge intermediate array if we were using np.interp, 
        # our fast_interp_uniform handles the array directly.
        
        # We need to call fast_interp_uniform for each row? 
        # No, we can pass the whole 2D array? 
        # fast_interp_uniform expects 1D x. 
        # We can reshape, call, and reshape back.
        
        quad_pos = log_grid[:, None] + log_ratio[None, :] # (n_grid, n_quad)
        quad_pos_flat = quad_pos.ravel()
        
        V_next_flat_interp = fast_interp_uniform(quad_pos_flat, log_S_min, step, V[i+1], K)
        V_next_interp = V_next_flat_interp.reshape(n_grid, n_quad)
        
        # Continuation value = disc * sum_j w_j * V_next_interp[k, j]
        cont = disc * (V_next_interp @ quad_weights)
        
        # Intrinsic value
        intrinsic = intrinsic_grid
        
        # Optimal value
        V[i] = np.maximum(intrinsic, cont)
        
        # Store continuation value for policy
        cont_values[i] = cont
        
        # Check time
        if time.time() > deadline - 10.0:
            # Fill remaining with intrinsic
            for j in range(i - 1, -1, -1):
                V[j] = intrinsic_grid
                if j < n_exercise - 1:
                     cont_values[j] = intrinsic_grid
            break

    # If we broke out early, ensure all cont_values are set
    for i in range(n_exercise - 1):
        if cont_values[i] is None:
            cont_values[i] = intrinsic_grid

    class ExercisePolicy:
        def __init__(self, cont_values, log_grid, step, K, n_grid):
            self.cont_values = cont_values
            self.log_grid = log_grid
            self.step = step
            self.K = K
            self.n_grid = n_grid
            self.log_S_min = log_grid[0]

        def exercise(self, current_state, time_idx):
            """
            Determine exercise decision for given states at a specific time index.
            """
            S = np.asarray(current_state, dtype=float)
            scalar_input = S.ndim == 0
            if scalar_input:
                S = S.reshape(1)
            
            n_states = S.shape[0]
            
            if time_idx < 0 or time_idx >= 12:
                return np.zeros(n_states, dtype=bool) if not scalar_input else False
            
            if time_idx == 11:
                # Last exercise date: exercise if in the money
                result = S < self.K
                return result[0] if scalar_input else result
            
            if self.cont_values[time_idx] is None:
                # Fallback: exercise if in the money
                result = S < self.K
                return result[0] if scalar_input else result
            
            # Interpolate continuation value
            log_S = np.log(S + 1e-15)
            
            # Use the same fast interpolation logic
            # We can reuse the function from the outer scope if we define it there, 
            # but to keep the class self-contained, we'll implement a small version or use np.interp 
            # for the query time which is small (8192 points).
            
            # For query time, np.interp is sufficient and simpler.
            # 8192 points is small.
            cont = np.interp(log_S, self.log_grid, self.cont_values[time_idx])
            
            intrinsic = np.maximum(self.K - S, 0.0)
            
            result = intrinsic >= cont
            
            return result[0] if scalar_input else result

    return ExercisePolicy(cont_values, log_grid, step, K, n_grid)
# EVOLVE-BLOCK-END
