import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12  # monthly exercise dates, no time-0
    
    # training_paths shape: (N_paths, n_time_points)
    # n_time_points = n_exercise + 1 = 13 (t=0, 1/12, ..., 1)
    N, n_tp = training_paths.shape
    dt = T / n_exercise
    discount = np.exp(-r * dt)
    
    # V[i, path] = value at exercise date i for path
    # Indices: 0..n_exercise
    # V[n_exercise] is maturity (col n_exercise)
    # V[i] is value at col i (for i=1..n_exercise)
    # We store V as (n_exercise+1, N)
    V = np.zeros((n_exercise + 1, N))
    
    # Maturity: must exercise if in the money
    V[n_exercise] = np.maximum(K - training_paths[:, n_exercise], 0)
    
    # Store thresholds for the policy
    # thresholds[i] corresponds to exercise date i+1 (col i+1)
    # thresholds[0] -> col 1, thresholds[10] -> col 11, thresholds[11] -> col 12
    thresholds = np.zeros(n_exercise)
    
    # Use a log-space grid for better resolution of put payoffs
    # This captures the asymmetry where low stock prices matter more for puts
    n_grid = 200
    # Grid from small positive value to well above K
    S_min_grid = max(1.0, K * 0.01)
    S_max_grid = K * 5.0
    log_S_min = np.log(S_min_grid)
    log_S_max = np.log(S_max_grid)
    log_grid = np.linspace(log_S_min, log_S_max, n_grid)
    S_grid = np.exp(log_grid)
    
    # Precompute exercise payoffs on the grid for each step
    # We'll store value functions on the grid
    # V_grid[k] = value at step k on the grid (size n_grid)
    V_grid = np.zeros((n_exercise + 1, n_grid))
    
    # Maturity value on grid
    V_grid[n_exercise] = np.maximum(K - S_grid, 0)
    
    for k in range(n_exercise - 1, 0, -1):
        if time.time() > deadline - 15:
            # Use simple heuristic for remaining steps
            for j in range(k, 0, -1):
                thresholds[j-1] = K * 0.8
            break
        
        S_k = training_paths[:, k]
        y = discount * V[k + 1]
        
        # Interpolate continuation values onto the grid
        # Use quantile-based binning on training data, then interpolate to grid
        n_bins = 150
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(S_k, quantiles)
        bin_edges = np.unique(bin_edges)
        
        if len(bin_edges) < 3:
            bin_edges = np.array([S_k.min(), (S_k.min() + S_k.max()) / 2, S_k.max()])
        
        # Assign each point to a bin
        bin_indices = np.searchsorted(bin_edges, S_k, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
        
        # Compute mean continuation value for each bin
        n_actual_bins = len(bin_edges) - 1
        cont_per_bin = np.zeros(n_actual_bins)
        
        for b in range(n_actual_bins):
            mask = bin_indices == b
            if np.any(mask):
                cont_per_bin[b] = np.mean(y[mask])
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Interpolate to the grid points
        # Handle points outside the range
        cont_on_grid = np.interp(S_grid, bin_centers, cont_per_bin,
                                 left=cont_per_bin[0], right=cont_per_bin[-1])
        cont_on_grid = np.maximum(cont_on_grid, 0)
        
        exercise_payoff_grid = np.maximum(K - S_grid, 0)
        V_grid[k] = np.maximum(exercise_payoff_grid, cont_on_grid)
        
        # Also compute V[k] on training paths for consistency
        cont_values = np.interp(S_k, bin_centers, cont_per_bin,
                                left=cont_per_bin[0], right=cont_per_bin[-1])
        cont_values = np.maximum(cont_values, 0)
        exercise_payoff = np.maximum(K - S_k, 0)
        V[k] = np.maximum(exercise_payoff, cont_values)
        
        # Determine threshold from the grid: largest S where exercise >= continuation
        should_exercise_grid = exercise_payoff_grid >= cont_on_grid
        if np.any(should_exercise_grid):
            last_true_idx = np.where(should_exercise_grid)[0][-1]
            thresholds[k-1] = S_grid[last_true_idx]
        else:
            thresholds[k-1] = 0.0
    
    # Set threshold for the last exercise date (maturity)
    thresholds[n_exercise - 1] = K
    
    # Create policy object
    class BermudanPutPolicy:
        def __init__(self, thresholds, n_exercise, K):
            self.thresholds = thresholds
            self.n_exercise = n_exercise
            self.K = K
        
        def exercise(self, current_state, time_step):
            # current_state: array of stock prices at current time
            # time_step: index into the time grid (integer)
            # Return boolean vector: True if exercise
            try:
                if isinstance(time_step, (int, np.integer)):
                    idx = int(time_step)
                else:
                    # If time_step is a float time, convert to index
                    idx = int(np.round(time_step * 12))
                
                # Exercise dates are at indices 1..n_exercise
                if idx < 1 or idx > self.n_exercise:
                    return np.zeros(len(current_state), dtype=bool)
                
                # Map to thresholds index
                thresh = self.thresholds[idx - 1]
                
                # Exercise if S <= threshold
                return current_state <= thresh
            except:
                return np.zeros(len(current_state), dtype=bool)
    
    return BermudanPutPolicy(thresholds, n_exercise, K)
