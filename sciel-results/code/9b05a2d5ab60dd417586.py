"""Bermudan exercise policy using binomial-tree backward induction with critical-price interpolation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    # 12 monthly exercise dates, no time-0 exercise
    n_steps = 12
    dt = T / n_steps
    
    # Build a binomial tree for the underlying
    # u = exp(sigma * sqrt(dt)), d = 1/u, p = (exp(r*dt) - d) / (u - d)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p_up = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    
    # We'll work with a grid of log-spaced stock prices
    # The tree has n_steps+1 levels (0 to n_steps)
    # At level k, stock can be S0 * u^j * d^(k-j) for j=0..k
    # But we don't know S0 exactly; the contract says spot 90/100/110 are tested.
    # Actually, the policy should work for any initial spot. 
    # The binomial tree approach typically fixes S0 and builds a tree.
    # But here, the exercise policy is called with current_state (array of spots) and time.
    # We need a policy that maps (S, t) -> exercise or not.
    
    # Since the model is GBM, the distribution of S_t given S_0 is known.
    # However, the exercise policy should be a function of the current state S and time t.
    # The binomial tree gives us critical prices: for each time step, there is a critical stock price S_crit such that:
    #   Exercise if S <= S_crit (for a put)
    #   Continue if S > S_crit
    #
    # We can compute these critical prices using a fine grid and backward induction.
    # Since the policy must work for any S0, we can assume a reference S0 (e.g., 100) and compute
    # the critical prices as fractions or in absolute terms. But the critical price depends on S0.
    #
    # Alternative: Use the fact that for GBM, the value function is homogeneous of degree 1 in S and K.
    # So V(S, t) = S * v(S/K, t) for some function v.
    # We can normalize by K. Let x = S/K. Then the problem depends only on x and t.
    # We can compute the critical x_crit at each time step, and then S_crit = K * x_crit.
    
    # Let's use a grid in log-space for x = S/K.
    # Range: x from 0.1 to 3.0 (S from 10 to 300)
    # Number of grid points: 500
    
    n_grid = 500
    x_min = 0.1
    x_max = 3.0
    x_grid = np.geomspace(x_min, x_max, n_grid)
    
    # Backward induction on the grid
    # V[k, i] = value at step k (0 = maturity, n_steps = today) and grid point i
    # At maturity (k=0), V[0, i] = max(K - S, 0) = K * max(1 - x, 0)
    
    # But we need to be careful with the time indexing.
    # Let's define:
    #   t_k = k * dt, for k = 0, 1, ..., n_steps
    #   Exercise dates are t_1, t_2, ..., t_n_steps
    #   Maturity is t_n_steps = T
    #
    # In backward induction, we start at t_n_steps (maturity) and go back to t_1.
    # At t_n_steps, the value is the intrinsic value (since no future exercise).
    # At t_{n_steps-1}, we compare intrinsic vs. continuation.
    # ...
    # At t_1, we compare intrinsic vs. continuation.
    
    # Let's index by the number of steps remaining.
    # m = number of steps remaining. m = 0 at maturity, m = n_steps at the first exercise date.
    # V[m, i] = value when m steps remain and stock is at x_grid[i].
    
    # At m = 0 (maturity):
    #   V[0, i] = K * max(1 - x_grid[i], 0)
    
    # For m = 1, 2, ..., n_steps:
    #   The stock can go up to x * u or down to x * d in the next step.
    #   Continuation value = disc * (p_up * V[m-1, j_up] + (1-p_up) * V[m-1, j_down])
    #   where j_up and j_down are the indices corresponding to x*u and x*d.
    #   Intrinsic value = K * max(1 - x, 0)
    #   V[m, i] = max(Intrinsic, Continuation)
    #   Exercise if Intrinsic > Continuation
    
    # We need to map x*u and x*d to grid indices.
    # Since the grid is geometric, x*u = x_grid[i] * u.
    # We can use searchsorted to find the nearest grid points.
    
    # Precompute the up and down neighbors
    # For each grid point i, find the indices for x_grid[i] * u and x_grid[i] * d
    
    # Since the grid is geometric, x_grid[i] * u = x_grid[i] * exp(sigma*sqrt(dt))
    # We can compute the log-grid: log_x = log(x_grid)
    # log_x_up = log_x + log(u)
    # log_x_down = log_x + log(d) = log_x - log(u)
    
    log_x = np.log(x_grid)
    log_u = np.log(u)
    
    # For each i, find the two nearest grid points for the up and down moves
    # and compute the interpolation weights
    
    # This is a bit complex. Let's use a simpler approach:
    # Use a uniform grid in log-space and interpolate.
    
    # Actually, let's use a different approach: compute the critical prices directly.
    # For each time step m, the critical price S_crit[m] is the stock price at which
    # the holder is indifferent between exercising and continuing.
    
    # We can compute this by solving for S such that:
    #   K - S = disc * E[V[m-1](S')]
    # where S' is the next stock price.
    
    # But this requires knowing V[m-1], which is a function.
    
    # Let's stick with the grid approach and compute V[m, i] for all m and i.
    # Then, for each m, find the critical x where V[m, i] switches from intrinsic to continuation.
    
    # Initialize V
    V = np.zeros((n_steps + 1, n_grid))
    
    # Maturity: m = 0
    V[0, :] = np.maximum(K - x_grid * K, 0.0)
    
    # Backward induction
    for m in range(1, n_steps + 1):
        if time.time() > DEADLINE:
            break
            
        # For each grid point, compute the continuation value
        # x_up = x_grid * u, x_down = x_grid * d
        # Find the nearest grid points and interpolate
        
        x_up = x_grid * u
        x_down = x_grid * d
        
        # Find indices for interpolation
        # Since the grid is sorted, we can use searchsorted
        idx_up_right = np.searchsorted(x_grid, x_up, side='right')
        idx_up_left = idx_up_right - 1
        
        idx_down_right = np.searchsorted(x_grid, x_down, side='right')
        idx_down_left = idx_down_right - 1
        
        # Clip indices
        idx_up_right = np.clip(idx_up_right, 1, n_grid - 1)
        idx_up_left = np.clip(idx_up_left, 0, n_grid - 2)
        
        idx_down_right = np.clip(idx_down_right, 1, n_grid - 1)
        idx_down_left = np.clip(idx_down_left, 0, n_grid - 2)
        
        # Interpolation weights
        w_up = (x_up - x_grid[idx_up_left]) / (x_grid[idx_up_right] - x_grid[idx_up_left] + 1e-10)
        w_down = (x_down - x_grid[idx_down_left]) / (x_grid[idx_down_right] - x_grid[idx_down_left] + 1e-10)
        
        # Clip weights to [0, 1]
        w_up = np.clip(w_up, 0.0, 1.0)
        w_down = np.clip(w_down, 0.0, 1.0)
        
        # Interpolated values
        V_up = (1 - w_up) * V[m-1, idx_up_left] + w_up * V[m-1, idx_up_right]
        V_down = (1 - w_down) * V[m-1, idx_down_left] + w_down * V[m-1, idx_down_right]
        
        # Continuation value
        cont_val = disc * (p_up * V_up + (1 - p_up) * V_down)
        
        # Intrinsic value
        intr_val = np.maximum(K - x_grid * K, 0.0)
        
        # Value
        V[m, :] = np.maximum(intr_val, cont_val)
    
    # Now, for each time step m (1 to n_steps), find the critical x.
    # The critical x is where the policy switches from exercise to continue.
    # For a put, exercise when x is small (stock is low).
    # So we look for the largest x such that V[m, i] == intrinsic (i.e., exercise is optimal).
    
    # Actually, we want the boundary: the largest x such that it's optimal to exercise.
    # Or equivalently, the smallest x such that it's optimal to continue.
    
    # Let's compute the exercise indicator for each grid point
    # exercise[m, i] = True if intrinsic >= continuation
    
    # Recompute to get the exercise decisions
    exercise_indicators = np.zeros((n_steps + 1, n_grid), dtype=bool)
    
    for m in range(1, n_steps + 1):
        if time.time() > DEADLINE:
            break
            
        x_up = x_grid * u
        x_down = x_grid * d
        
        idx_up_right = np.searchsorted(x_grid, x_up, side='right')
        idx_up_left = idx_up_right - 1
        idx_down_right = np.searchsorted(x_grid, x_down, side='right')
        idx_down_left = idx_down_right - 1
        
        idx_up_right = np.clip(idx_up_right, 1, n_grid - 1)
        idx_up_left = np.clip(idx_up_left, 0, n_grid - 2)
        idx_down_right = np.clip(idx_down_right, 1, n_grid - 1)
        idx_down_left = np.clip(idx_down_left, 0, n_grid - 2)
        
        w_up = (x_up - x_grid[idx_up_left]) / (x_grid[idx_up_right] - x_grid[idx_up_left] + 1e-10)
        w_down = (x_down - x_grid[idx_down_left]) / (x_grid[idx_down_right] - x_grid[idx_down_left] + 1e-10)
        w_up = np.clip(w_up, 0.0, 1.0)
        w_down = np.clip(w_down, 0.0, 1.0)
        
        V_up = (1 - w_up) * V[m-1, idx_up_left] + w_up * V[m-1, idx_up_right]
        V_down = (1 - w_down) * V[m-1, idx_down_left] + w_down * V[m-1, idx_down_right]
        
        cont_val = disc * (p_up * V_up + (1 - p_up) * V_down)
        intr_val = np.maximum(K - x_grid * K, 0.0)
        
        exercise_indicators[m, :] = intr_val >= cont_val
    
    # For each m, find the critical x
    # critical_x[m] = the largest x_grid[i] such that exercise_indicators[m, i] is True
    # If no such x exists, critical_x[m] = 0 (always continue)
    # If all are True, critical_x[m] = x_grid[-1] (always exercise)
    
    critical_x = np.zeros(n_steps + 1)
    for m in range(1, n_steps + 1):
        if time.time() > DEADLINE:
            break
            
        # Find the last index where exercise is True
        # Since exercise is True for small x and False for large x (for a put)
        # We can find the boundary
        ex = exercise_indicators[m, :]
        
        # Find the last True
        true_indices = np.where(ex)[0]
        if len(true_indices) > 0:
            critical_x[m] = x_grid[true_indices[-1]]
        else:
            critical_x[m] = 0.0
    
    # Now we have critical_x[m] for m = 1, ..., n_steps
    # The policy: at time t with m steps remaining, exercise if S <= K * critical_x[m]
    # m = round((T - t) / dt) = round((T - t) * 12 / T) = round(12 - 12*t/T)
    # Since T=1, m = round(12 - 12*t)
    
    class ExercisePolicy:
        def __init__(self):
            self.critical_x = critical_x
            self.K = K
            self.n_steps = n_steps
            self.T = T
            self.dt = dt
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: current time
            Returns: boolean array
            """
            S = np.asarray(current_state).flatten()
            
            # Compute m: number of steps remaining
            # t is in [0, T]
            # m = round((T - time) / dt)
            # Clamp to [0, n_steps]
            m_float = (self.T - time) / self.dt
            m = int(round(m_float))
            m = max(0, min(m, self.n_steps))
            
            if m == 0:
                # Maturity: exercise if intrinsic > 0
                return np.maximum(self.K - S, 0.0) > 0.0
            
            # Get critical price
            S_crit = self.K * self.critical_x[m]
            
            # Exercise if S <= S_crit
            return S <= S_crit
    
    policy = ExercisePolicy()
    return policy

# EVOLVE-BLOCK-END
