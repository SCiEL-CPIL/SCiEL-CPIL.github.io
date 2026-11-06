import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_ex = contract.get('n_exercise_dates', 12)
    
    # Exercise times: monthly, no time-0
    dt = T / n_ex
    exercise_times = np.arange(1, n_ex + 1) * dt
    
    # Ensure training_paths is a 2D array
    training_paths = np.asarray(training_paths)
    if training_paths.ndim == 1:
        training_paths = training_paths.reshape(-1, 1)
    
    n_paths, n_steps = training_paths.shape
    
    # Risk-neutral parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p_up = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    
    # Terminal payoff (European put at T)
    terminal_payoff = np.maximum(K - training_paths[:, -1], 0.0)
    
    # For each exercise date, we'll learn a threshold using regression
    # We use the training paths to estimate continuation values
    
    # Create a fine grid for interpolation
    S_min = max(1.0, K * 0.1)
    S_max = K * 3.0
    n_grid = 5000
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # Initialize continuation value function at terminal
    # V_next[S] = max(K - S, 0)
    V_next_grid = np.maximum(K - S_grid, 0.0)
    
    # Thresholds for each exercise date
    thresholds = np.zeros(n_ex)
    
    # Backward induction with regression on training data
    for k in range(n_ex - 1, -1, -1):
        if time.time() > deadline - 5.0:
            break
        
        # Get the stock prices at exercise date k for all training paths
        # Exercise dates are at indices 1, 2, ..., 12 (0-indexed: 1 to 12)
        # So at exercise date k (0-indexed), the price index is k+1
        S_at_ex = training_paths[:, k + 1]
        
        # Exercise value
        ex_val = np.maximum(K - S_at_ex, 0.0)
        
        # Estimate continuation value using regression
        # We'll use a simple approach: fit a polynomial or use the grid-based value
        # For speed, use the grid-based continuation value interpolated at S_at_ex
        
        # Compute continuation value on the grid
        S_up = S_grid * u
        S_down = S_grid * d
        
        V_up = np.interp(S_up, S_grid, V_next_grid, left=0.0, right=0.0)
        V_down = np.interp(S_down, S_grid, V_next_grid, left=0.0, right=0.0)
        
        cont_val_grid = disc * (p_up * V_up + (1 - p_up) * V_down)
        
        # Interpolate continuation value at the training path prices
        cont_val = np.interp(S_at_ex, S_grid, cont_val_grid, left=0.0, right=0.0)
        
        # Determine optimal exercise: exercise if ex_val > cont_val
        exercise_mask = ex_val > cont_val
        
        # Find the threshold: the largest S where exercise is optimal
        # Sort by S and find the boundary
        if np.any(exercise_mask):
            # Get the maximum S where exercise is optimal
            S_exercised = S_at_ex[exercise_mask]
            if len(S_exercised) > 0:
                # Use a quantile to be robust to outliers
                thresholds[k] = np.percentile(S_exercised, 99)
            else:
                thresholds[k] = 0.0
        else:
            thresholds[k] = 0.0
        
        # Update V_next_grid: max(ex_val, cont_val) on the grid
        ex_val_grid = np.maximum(K - S_grid, 0.0)
        V_next_grid = np.maximum(ex_val_grid, cont_val_grid)
    
    # Create policy object
    class BermudanPolicy:
        def __init__(self, thresholds, exercise_times, K, r, sigma, T, n_ex):
            self.thresholds = thresholds
            self.exercise_times = exercise_times
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_ex = n_ex
        
        def exercise(self, current_state, time_val):
            """
            current_state: array of stock prices
            time_val: current time (scalar)
            returns: boolean array, True if exercise
            """
            current_state = np.asarray(current_state)
            
            # Find which exercise date we're at
            # Check if time_val is close to an exercise date
            for k in range(self.n_ex):
                if abs(time_val - self.exercise_times[k]) < 1e-4:
                    return current_state <= self.thresholds[k]
            
            # Not at an exercise date, cannot exercise
            return np.zeros(len(current_state), dtype=bool)
        
        def __call__(self, current_state, time_val):
            return self.exercise(current_state, time_val)
    
    policy = BermudanPolicy(thresholds, exercise_times, K, r, sigma, T, n_ex)
    return policy
