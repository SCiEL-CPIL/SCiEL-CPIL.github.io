"""Bermudan put exercise policy: high-fidelity backward induction with adaptive grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class ExercisePolicy:
    def __init__(self, thresholds, grid, K):
        self.thresholds = thresholds  # List of arrays, shape (n_grid,) for each exercise date
        self.grid = grid              # Log-spaced spot price grid
        self.K = K
        self.log_grid = np.log(grid)
        self.n_grid = len(grid)
        
    def exercise(self, current_state, time_idx):
        """
        Determine exercise decision for given states at a specific time index.
        
        Parameters:
        -----------
        current_state : array-like
            Spot prices for a batch of paths.
        time_idx : int
            The exercise date index (1-based, 1..12). If 0, never exercise.
            
        Returns:
        --------
        boolean array
            True if exercise is optimal for that state/time.
        """
        if time_idx <= 0 or time_idx > len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
        
        states = np.asarray(current_state, dtype=np.float64)
        # Clip states to grid range for safety, though interpolation is not used here 
        # because we use nearest-neighbor or direct lookup logic? 
        # Actually, for robustness, we should interpolate the threshold or just check 
        # if S < Threshold. 
        # The optimal policy for a put is S < S*. 
        # We have computed the continuation value C(S) and intrinsic I(S).
        # Exercise if I(S) > C(S).
        # Since I(S) is linear and C(S) is convex/smooth, the boundary is a single point S*.
        # However, to be safe and general, we can just evaluate the condition directly 
        # if we stored the value functions, but we only stored thresholds?
        # Wait, the inspiration stored `cont_at_date`. 
        # Let's store the continuation values and intrinsic values, or just the boundary S*.
        # Storing the boundary S* is most efficient.
        
        # Let's re-verify the storage. 
        # In the inspiration, it calculated `cont_at_date[k_idx]` which is the continuation value.
        # And `V_next` is the value at the next step.
        # The condition is: Exercise if Payoff(S) > Continuation(S).
        # Payoff(S) = max(K-S, 0).
        # So we need to compare K-S with Cont(S).
        
        # For efficiency in the `exercise` call, we should precompute the threshold S* 
        # for each time step, or store the full continuation function and evaluate it.
        # Evaluating the full function requires interpolation.
        # Finding S* is O(n_grid) once.
        
        # Let's assume `self.thresholds` actually stores the critical spot price S* 
        # for each time step, or a function to determine exercise.
        # Given the grid is fine (2048+), we can find S* during fitting.
        
        # Let's refine: Store `exercise_thresholds` where thresholds[t] is S*.
        # Then exercise if S < S*.
        
        # But wait, is the region always S < S*? 
        # For a put, yes. Intrinsic value decreases with S. Continuation value increases with S (usually).
        # So there is a unique crossing point.
        
        # If we stored the continuation values array `cont_vals` and the grid, 
        # we can do:
        # intrinsic = np.maximum(self.K - states, 0)
        # continuation = np.interp(states, self.grid, self.cont_vals[time_idx-1])
        # return intrinsic > continuation
        
        # This is safer than assuming a single threshold if the grid is coarse or non-monotonic 
        # (though it should be monotonic). Interpolation is fast enough for 8192 paths.
        
        # Let's stick to the interpolation approach for robustness.
        # So `self.thresholds` should be renamed `self.cont_values` in the implementation below.
        pass

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on a log-grid.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s kill, leave 10s buffer)

    # Extract contract parameters
    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))
    n_steps = int(contract.get('n_steps', 12))

    dt = T / n_steps
    disc = np.exp(-r * dt)
    mu = r - 0.5 * sigma * sigma
    vol_step = sigma * np.sqrt(dt)

    # Grid construction
    # Use a wide range to cover all likely paths
    min_s = max(1.0, 10.0)  # Put is worthless if S is very high, but need low S for value
    max_s = max(K * 5.0, 500.0)
    
    # Adaptive grid size based on remaining time? 
    # Let's start with a fixed high resolution. 4096 points is good.
    n_grid = 4096
    log_min = np.log(min_s)
    log_max = np.log(max_s)
    log_grid = np.linspace(log_min, log_max, n_grid)
    grid = np.exp(log_grid)

    # Quadrature for expectation
    # Gauss-Hermite quadrature with 64 points
    n_quad = 64
    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_quad)
    z_quad = np.sqrt(2.0) * x_gh
    w_quad = w_gh / np.sqrt(np.pi)

    # Precompute transition matrix? 
    # S_next = S * exp(mu*dt + vol_step*z)
    # log_S_next = log_S + mu*dt + vol_step*z
    # We need E[V_next(S_next) | S]
    # This is sum_j w_j V_next(exp(log_S + shift_j))
    
    # We can vectorize this using matrix multiplication if we precompute the shifts?
    # log_S_next[i, j] = log_grid[i] + mu*dt + vol_step*z_quad[j]
    # This creates a (n_grid, n_quad) array.
    # Then we need to evaluate V_next at these points.
    # Since V_next is defined on `grid` (log_grid), we can use np.interp.
    # But doing np.interp for each row is slow.
    # Better: Use the fact that the grid is log-uniform and the shift is additive in log-space.
    # If we represent V_next on the log-grid, the expectation is a convolution in log-space?
    # No, because the weight function is Gaussian in log-space (since Z is normal, log S_next is normal).
    # So E[V_next(S_next)] = integral V_next(exp(x)) * N(x; log(S) + mu*dt, vol_step^2) dx
    # This is a Gaussian blur of the function f(x) = V_next(exp(x)) with std dev vol_step.
    # We can implement this via FFT convolution or simple weighted sum.
    # Given n_quad=64 and n_grid=4096, a simple loop over quadrature points with vectorized interpolation
    # might be okay, but 4096 * 64 * 12 steps = ~3M ops. 
    # Let's try to optimize the inner loop.
    
    # Strategy:
    # 1. Represent V on the log-grid.
    # 2. For each step, compute V_cont[i] = sum_j w_j * V_next[interp(log_grid[i] + mu*dt + vol_step*z_j)]
    
    # To speed up interpolation, we can precompute the indices? 
    # Or use a faster interpolation method.
    # Since the grid is uniform in log-space, we can use linear interpolation manually.
    
    def fast_interp_log(log_vals, log_grid, values):
        """
        Linearly interpolate values at log_vals using log_grid.
        log_vals: array of log prices
        log_grid: 1D array of grid log prices
        values: 1D array of values on log_grid
        """
        # Clip to range
        min_log = log_grid[0]
        max_log = log_grid[-1]
        log_vals_clipped = np.clip(log_vals, min_log, max_log)
        
        # Find indices
        # Since log_grid is sorted, we can use searchsorted
        idx = np.searchsorted(log_grid, log_vals_clipped)
        # Handle edge cases
        idx = np.clip(idx, 1, len(log_grid) - 1)
        
        x0 = log_grid[idx - 1]
        x1 = log_grid[idx]
        y0 = values[idx - 1]
        y1 = values[idx]
        
        # Linear interpolation
        t = (log_vals_clipped - x0) / (x1 - x0)
        return y0 * (1 - t) + y1 * t

    # Backward Induction
    # V_next: value function at the next exercise date, on the grid
    # At the last exercise date (k=n_steps), V = Payoff
    V_next = np.maximum(K - grid, 0.0)
    
    # Store continuation values for each date (1 to n_steps)
    # cont_values[k-1] will be the continuation value at date k
    # Note: At date n_steps, continuation is 0 (no further dates).
    # But we need the policy at all dates.
    # Policy at date k: Exercise if Payoff(S) > Cont(S)
    # Where Cont(S) = disc * E[V_{k+1}(S_{k+1}) | S_k]
    
    cont_values_list = []
    
    # We iterate backwards from k = n_steps-1 down to 1
    # At k = n_steps, we don't compute a continuation for the *next* step because it doesn't exist.
    # The value at n_steps is just the payoff.
    # The decision at n_steps: Exercise if Payoff > 0 (since no continuation).
    # So for date n_steps, Cont = 0.
    
    # Let's store Cont for each date to use in the exercise() method.
    # cont_values_list will have length n_steps.
    # cont_values_list[k-1] is the continuation value at date k.
    
    # Initialize for the last date
    # At date n_steps, continuation is 0.
    cont_values_list.append(np.zeros(n_grid)) # Index n_steps-1 corresponds to date n_steps
    
    # Now go backwards from k = n_steps-1 to 1
    # V_next is currently V_{n_steps}
    # We need to compute V_k = max(Payoff, disc * E[V_{k+1}])
    # And store the continuation part: Cont_k = disc * E[V_{k+1}]
    
    for k in range(n_steps - 1, 0, -1):
        # Check time
        if time.time() > deadline:
            # Fallback: return a simple heuristic if time runs out
            # This shouldn't happen with 4096 grid and 64 quad points
            break
        
        # Compute E[V_next(S_next) | S]
        # For each grid point i, we need sum_j w_j * V_next(exp(log_grid[i] + mu*dt + vol_step*z_j))
        
        # Vectorized approach:
        # Create a matrix of log_next values: shape (n_grid, n_quad)
        # log_next[i, j] = log_grid[i] + mu*dt + vol_step*z_quad[j]
        
        # To avoid creating a huge (4096, 64) array every time and doing expensive searchsorted,
        # we can exploit the structure.
        # However, 4096*64 = 262k elements. This is small.
        
        shift = mu * dt
        log_next_matrix = log_grid[:, np.newaxis] + shift + vol_step * z_quad[np.newaxis, :]
        
        # Now we need to evaluate V_next at log_next_matrix.
        # V_next is defined on log_grid.
        # We can use the fast_interp_log function, but it expects 1D.
        # Let's flatten, interpolate, and reshape.
        
        log_next_flat = log_next_matrix.flatten()
        
        # Optimization: Since log_grid is uniform, we can compute the interpolation weights more directly.
        # But let's try the searchsorted approach first.
        # If it's too slow, we'll switch to FFT or precomputed indices.
        
        # Actually, for uniform grids, we can compute the fractional index directly.
        # step = log_grid[1] - log_grid[0]
        # idx_float = (log_next_flat - log_grid[0]) / step
        # idx0 = floor(idx_float)
        # t = idx_float - idx0
        
        step = log_grid[1] - log_grid[0]
        
        # Handle boundaries
        idx_float = (log_next_flat - log_grid[0]) / step
        idx_float = np.clip(idx_float, 0, n_grid - 1)
        
        idx0 = idx_float.astype(int)
        idx1 = np.clip(idx0 + 1, 0, n_grid - 1)
        t = idx_float - idx0
        
        # Clip indices to valid range
        idx0 = np.clip(idx0, 0, n_grid - 1)
        idx1 = np.clip(idx1, 0, n_grid - 1)
        
        y0 = V_next[idx0]
        y1 = V_next[idx1]
        
        V_next_at_next = y0 * (1 - t) + y1 * t
        
        # Reshape back
        V_next_at_next = V_next_at_next.reshape(n_grid, n_quad)
        
        # Weighted sum
        # E_val[i] = sum_j w_quad[j] * V_next_at_next[i, j]
        E_val = np.dot(V_next_at_next, w_quad)
        
        Cont = disc * E_val
        
        # Store continuation value for this date k
        # k is 1-based in the loop? No, k is from n_steps-1 down to 1.
        # So k corresponds to date k.
        # We need to insert this into the list at the correct position.
        # The list currently has cont for date n_steps at index n_steps-1.
        # We are computing cont for date k.
        # So we append to the end and then reverse? Or insert at k-1?
        # Let's just create a list of size n_steps and fill it.
        
        if len(cont_values_list) < n_steps:
             # This logic is getting messy. Let's pre-allocate.
             pass
        
        # Let's restart the storage logic to be cleaner.
        # We'll use a list `conts` of length n_steps.
        # conts[k-1] is the continuation value at date k.
        
        # Re-initialize storage
        # I will restructure the loop to be cleaner in the final code.
        
        # For now, let's assume we are filling a pre-allocated list.
        # In the actual code below, I will handle this properly.
        
        # Update V_next for the next iteration (going further back)
        # V_k = max(Payoff, Cont)
        Payoff = np.maximum(K - grid, 0.0)
        V_next = np.maximum(Payoff, Cont)

    # The loop above was a sketch. Here is the proper implementation structure.
    
    # --- Proper Implementation ---
    
    conts = [None] * n_steps
    # Last date (index n_steps-1)
    conts[n_steps-1] = np.zeros(n_grid)
    
    V_next = np.maximum(K - grid, 0.0)
    
    step = log_grid[1] - log_grid[0]
    
    for k in range(n_steps - 1, 0, -1):
        if time.time() > deadline:
            # Fill remaining with zeros or heuristic if timeout
            for rem_k in range(k-1, 0, -1):
                conts[rem_k-1] = np.zeros(n_grid)
            break
            
        shift = mu * dt
        # log_next[i, j] = log_grid[i] + shift + vol_step * z_quad[j]
        # We can compute this as:
        # log_grid[:, None] + shift + vol_step * z_quad[None, :]
        
        log_next = log_grid[:, np.newaxis] + shift + vol_step * z_quad[np.newaxis, :]
        
        # Interpolate V_next at log_next
        # Flatten
        log_next_flat = log_next.flatten()
        
        # Compute indices
        idx_float = (log_next_flat - log_grid[0]) / step
        idx_float = np.clip(idx_float, 0, n_grid - 1)
        
        idx0 = idx_float.astype(np.int64)
        idx1 = np.clip(idx0 + 1, 0, n_grid - 1)
        t = idx_float - idx0
        
        # Clip idx0 to be safe (though idx_float is clipped, floor might be n_grid-1)
        idx0 = np.clip(idx0, 0, n_grid - 1)
        
        y0 = V_next[idx0]
        y1 = V_next[idx1]
        
        V_next_at_next = y0 * (1 - t) + y1 * t
        V_next_at_next = V_next_at_next.reshape(n_grid, n_quad)
        
        E_val = np.dot(V_next_at_next, w_quad)
        Cont = disc * E_val
        
        conts[k-1] = Cont
        
        Payoff = np.maximum(K - grid, 0.0)
        V_next = np.maximum(Payoff, Cont)

    # If timeout occurred in the middle, some conts might be None. Fill them.
    for i in range(n_steps):
        if conts[i] is None:
            conts[i] = np.zeros(n_grid)

    # Create the policy object
    # We need to pass conts, grid, K
    # The exercise method will interpolate the continuation value for the given states
    # and compare with intrinsic value.
    
    class FinalPolicy:
        def __init__(self, conts, grid, log_grid, K, step):
            self.conts = conts
            self.grid = grid
            self.log_grid = log_grid
            self.K = K
            self.step = step
            self.n_grid = len(grid)
            
        def exercise(self, current_state, time_idx):
            if time_idx <= 0 or time_idx > len(self.conts):
                return np.zeros(len(current_state), dtype=bool)
            
            states = np.asarray(current_state, dtype=np.float64)
            log_states = np.log(np.maximum(states, 1e-10))
            
            # Interpolate continuation value
            cont_vals = self.conts[time_idx - 1]
            
            idx_float = (log_states - self.log_grid[0]) / self.step
            idx_float = np.clip(idx_float, 0, self.n_grid - 1)
            
            idx0 = idx_float.astype(np.int64)
            idx1 = np.clip(idx0 + 1, 0, self.n_grid - 1)
            idx0 = np.clip(idx0, 0, self.n_grid - 1)
            
            t = idx_float - idx0
            
            c0 = cont_vals[idx0]
            c1 = cont_vals[idx1]
            
            continuation = c0 * (1 - t) + c1 * t
            
            intrinsic = np.maximum(self.K - states, 0.0)
            
            # Exercise if intrinsic > continuation
            return intrinsic > continuation

    return FinalPolicy(conts, grid, log_grid, K, step)

# EVOLVE-BLOCK-END
