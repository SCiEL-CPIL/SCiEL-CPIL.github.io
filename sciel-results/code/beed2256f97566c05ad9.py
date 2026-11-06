"""Bermudan exercise-policy: Regression-based backward induction using training paths."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Parse contract parameters
    S0_list = contract.get('S0_list', [90.0, 100.0, 110.0])
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Exercise times: T/12, 2T/12, ..., T (no time-0 exercise)
    dt = T / n_dates
    exercise_times = np.array([i * dt for i in range(1, n_dates + 1)])
    
    # Ensure training_paths is a numpy array
    training_paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, n_steps = training_paths.shape
    
    # We need values at each exercise date
    # training_paths[i, j] = spot at step j for path i
    # Steps 0..n_steps-1 correspond to times 0, dt, 2dt, ..., n_steps*dt
    # Exercise dates are at indices 1, 2, ..., n_dates (assuming n_steps >= n_dates)
    
    # If n_steps < n_dates, we need to handle differently
    # Assume n_steps >= n_dates for now
    
    # For each exercise date j (1-indexed, so index j in 0-based), 
    # we have the spot price S_j and the continuation value
    # We want to learn: exercise if K - S_j > disc * E[continuation | S_j]
    
    # Approach: Backward induction with regression
    # Start from the last exercise date (n_dates)
    # At the last date, it's optimal to exercise if K - S > 0 (since no more exercise opportunities)
    # Wait, actually at the last exercise date, if you don't exercise, the option expires worthless
    # So at the last date: value = max(K - S, 0)
    
    # For earlier dates, we need to estimate the continuation value
    # Continuation value = disc * E[max(K - S_next, continuation_next) | S_current]
    
    # We'll use the training paths to estimate this via regression
    # For each date j, we fit a model that maps S_j to the continuation value
    
    # Let's work with the exercise dates only
    # Date indices: 1, 2, ..., n_dates (0-based in the array)
    
    # Initialize continuation values at the last exercise date
    # At date n_dates: cont_val = max(K - S, 0) (since exercising is always optimal if in-the-money)
    # Actually, at the last date, you MUST decide: exercise or let expire
    # So value at last date = max(K - S, 0)
    
    # For dates n_dates-1 down to 1:
    # We need to estimate E[value at date j+1 | S at date j]
    
    # Let's extract the relevant columns
    # Date j corresponds to index j in the training_paths array (0-based)
    # So date 1 is index 1, date 2 is index 2, ..., date n_dates is index n_dates
    
    # Check if we have enough steps
    if n_steps < n_dates:
        # Fallback: use binomial tree
        return _binomial_tree_policy(S0_list, K, r, sigma, T, n_dates)
    
    # Extract spot prices at exercise dates
    # S_j for j = 1, ..., n_dates
    S_at_dates = training_paths[:, 1:n_dates+1]  # shape (n_paths, n_dates)
    
    # Initialize: at the last exercise date (index n_dates-1 in S_at_dates), 
    # the value is max(K - S, 0)
    # We'll store the continuation value (value if you DON'T exercise) for each date
    
    # Actually, let's think about this differently:
    # At each date j, the total value = max(K - S_j, cont_val_j)
    # where cont_val_j = disc * E[total_value_{j+1} | S_j]
    
    # We want to learn a function f_j(S_j) = cont_val_j
    
    # Start from the last date:
    # At date n_dates: total_value = max(K - S, 0)
    # So cont_val at date n_dates is 0 (no future dates)
    # But we need cont_val for dates 1 to n_dates-1
    
    # Let's iterate backward
    # For date j (from n_dates-1 down to 1):
    #   We know total_value_{j+1} for all paths (computed from date j+1)
    #   We want to estimate E[total_value_{j+1} | S_j]
    #   We can use regression: fit a model that predicts total_value_{j+1} from S_j
    
    # We'll use a simple approach: for each date j, we'll use the training paths
    # to estimate the conditional expectation via binning or regression
    
    # Let's use a piecewise linear regression approach:
    # For each date j, we'll fit a function g_j(S) such that g_j(S_j) ≈ E[total_value_{j+1} | S_j]
    
    # To make this fast, we'll use a simple method:
    # 1. Sort paths by S_j
    # 2. For each S_j, compute the average total_value_{j+1} of paths with similar S_j
    
    # Actually, let's use a more robust approach:
    # We'll compute the continuation value at each date using the training paths
    # and a simple regression (e.g., linear or quadratic in log(S))
    
    # Let's use the following approach:
    # For each date j, we'll compute the continuation value for each path
    # by taking the average of total_value_{j+1} over paths with similar S_j
    
    # To make this efficient, we'll use a binning approach:
    # Divide the range of S_j into bins, and for each bin, compute the average total_value_{j+1}
    
    # Let's implement this:
    
    # Initialize total_value for the last date
    total_value = np.zeros((n_paths, n_dates))
    total_value[:, n_dates-1] = np.maximum(K - S_at_dates[:, n_dates-1], 0.0)
    
    # For each date from n_dates-2 down to 0 (corresponding to dates n_dates-1 down to 1):
    for j in range(n_dates - 2, -1, -1):
        # We want to estimate E[total_value[:, j+1] | S_at_dates[:, j]]
        S_j = S_at_dates[:, j]
        TV_next = total_value[:, j+1]
        
        # Use binning to estimate the conditional expectation
        # Choose number of bins based on n_paths
        n_bins = min(50, max(10, int(np.sqrt(n_paths))))
        
        # Create bins
        S_min = np.min(S_j)
        S_max = np.max(S_j)
        bin_edges = np.linspace(S_min, S_max, n_bins + 1)
        
        # Assign each path to a bin
        bin_indices = np.digitize(S_j, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute the average TV_next for each bin
        bin_sums = np.bincount(bin_indices, weights=TV_next, minlength=n_bins)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_avgs = np.zeros(n_bins)
        nonzero = bin_counts > 0
        bin_avgs[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]
        # For empty bins, use the overall average
        if not np.all(nonzero):
            bin_avgs[~nonzero] = np.mean(TV_next)
        
        # Interpolate the continuation value for each path
        # The center of each bin
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # For each path, find its S_j and interpolate
        # Use np.interp for linear interpolation
        cont_val_j = np.interp(S_j, bin_centers, bin_avgs)
        
        # Discount
        disc = np.exp(-r * dt)
        cont_val_j = disc * cont_val_j
        
        # Total value at date j
        total_value[:, j] = np.maximum(K - S_j, cont_val_j)
    
    # Now we have the total value at each date for each training path
    # We also have the continuation value (before taking max) at each date
    # We need to determine the exercise policy: exercise if K - S_j >= cont_val_j
    
    # Let's recompute the continuation values more carefully
    # We need to store cont_val_j for each date and path
    
    cont_vals = np.zeros((n_paths, n_dates))
    cont_vals[:, n_dates-1] = 0.0  # No continuation at the last date
    
    # Recompute backward
    for j in range(n_dates - 2, -1, -1):
        S_j = S_at_dates[:, j]
        TV_next = total_value[:, j+1]
        
        n_bins = min(50, max(10, int(np.sqrt(n_paths))))
        S_min = np.min(S_j)
        S_max = np.max(S_j)
        bin_edges = np.linspace(S_min, S_max, n_bins + 1)
        bin_indices = np.digitize(S_j, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_sums = np.bincount(bin_indices, weights=TV_next, minlength=n_bins)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_avgs = np.zeros(n_bins)
        nonzero = bin_counts > 0
        bin_avgs[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]
        if not np.all(nonzero):
            bin_avgs[~nonzero] = np.mean(TV_next)
        
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        cont_val_j = np.interp(S_j, bin_centers, bin_avgs)
        disc = np.exp(-r * dt)
        cont_val_j = disc * cont_val_j
        
        cont_vals[:, j] = cont_val_j
        total_value[:, j] = np.maximum(K - S_j, cont_val_j)
    
    # Now, for each date j, we have:
    # S_j: spot price
    # cont_vals[:, j]: continuation value
    # Exercise if K - S_j >= cont_vals[:, j]
    
    # We want to learn a threshold function: for each date j, find a threshold T_j
    # such that we exercise if S_j <= T_j
    
    # To find the threshold, we can look at the training data:
    # For each date j, find the S_j values where K - S_j = cont_vals[:, j]
    # This gives us the boundary
    
    # Let's fit a threshold for each date:
    # We'll use the fact that the optimal policy is monotone: exercise if S <= threshold_j
    
    # For each date j, we can find the threshold by looking at the training paths:
    # Sort by S_j, and find where K - S_j crosses cont_vals[:, j]
    
    thresholds = np.zeros(n_dates)
    for j in range(n_dates):
        S_j = S_at_dates[:, j]
        intrinsic = K - S_j
        cont = cont_vals[:, j]
        
        # Exercise if intrinsic >= cont
        # Find the boundary: S_j such that K - S_j = cont
        # i.e., S_j = K - cont
        
        # For each path, the boundary is K - cont
        # We want a single threshold: the median or some robust estimate
        
        # Let's use the median of (K - cont) for paths where this is meaningful
        boundary = K - cont
        
        # Filter to reasonable values
        valid = (boundary > 0) & (boundary < 2 * K)
        if np.sum(valid) > 0:
            thresholds[j] = np.median(boundary[valid])
        else:
            thresholds[j] = K  # Default
    
    # Now we have thresholds for each date
    # The policy is: at date j, exercise if S <= thresholds[j]
    
    # But we need to return an object that can be called with (current_state, time)
    # current_state is an array of spot prices
    # time is the current time
    
    # We need to map time to the exercise date index
    # Exercise dates are at times: dt, 2dt, ..., n_dates*dt
    
    class BermudanExercisePolicy:
        def __init__(self, thresholds, exercise_times, K):
            self.thresholds = thresholds
            self.exercise_times = exercise_times
            self.K = K
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: current time (scalar)
            Returns: boolean array indicating which paths should exercise
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            
            # Find the nearest exercise date
            # time should be one of the exercise_times
            # Find the index of the exercise date closest to time
            idx = np.argmin(np.abs(self.exercise_times - time))
            
            # Exercise if S <= threshold
            return current_state <= self.thresholds[idx]
    
    return BermudanExercisePolicy(thresholds, exercise_times, K)


def _binomial_tree_policy(S0_list, K, r, sigma, T, n_dates):
    """Fallback: simple binomial tree policy"""
    dt = T / n_dates
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    
    # Create a grid of possible spot prices
    # We'll use a log grid
    S_min = 1.0
    S_max = 3.0 * K
    n_grid = 200
    S_grid = np.logspace(np.log10(S_min), np.log10(S_max), n_grid)
    
    # Initialize value at maturity
    V = np.maximum(K - S_grid, 0.0)
    
    # Backward induction
    for j in range(n_dates - 1, -1, -1):
        # Compute continuation value
        V_up = V[np.clip(np.arange(n_grid) + 1, 0, n_grid - 1)]
        V_down = V[np.clip(np.arange(n_grid) - 1, 0, n_grid - 1)]
        cont = disc * (p * V_up + (1 - p) * V_down)
        V = np.maximum(K - S_grid, cont)
    
    # Now V is the value at time 0 for each grid point
    # But we need the exercise policy at each date
    
    # Let's recompute to get the policy at each date
    V = np.maximum(K - S_grid, 0.0)
    policies = []
    
    for j in range(n_dates - 1, -1, -1):
        V_up = V[np.clip(np.arange(n_grid) + 1, 0, n_grid - 1)]
        V_down = V[np.clip(np.arange(n_grid) - 1, 0, n_grid - 1)]
        cont = disc * (p * V_up + (1 - p) * V_down)
        exercise = (K - S_grid) >= cont
        policies.append(exercise)
        V = np.maximum(K - S_grid, cont)
    
    # Reverse policies to get them in chronological order
    policies.reverse()
    
    # For each date, find the threshold
    thresholds = np.zeros(n_dates)
    for j in range(n_dates):
        ex = policies[j]
        if np.any(ex) and np.any(~ex):
            # Find the boundary
            idx = np.where(ex & ~np.roll(ex, 1))[0]
            if len(idx) > 0:
                thresholds[j] = S_grid[idx[0]]
            else:
                thresholds[j] = S_grid[np.max(np.where(ex)[0])] if np.any(ex) else 0
        elif np.all(ex):
            thresholds[j] = S_max
        else:
            thresholds[j] = 0
    
    exercise_times = np.array([i * dt for i in range(1, n_dates + 1)])
    
    class BermudanExercisePolicy:
        def __init__(self, thresholds, exercise_times, K):
            self.thresholds = thresholds
            self.exercise_times = exercise_times
            self.K = K
            
        def exercise(self, current_state, time):
            current_state = np.asarray(current_state, dtype=np.float64)
            idx = np.argmin(np.abs(self.exercise_times - time))
            return current_state <= self.thresholds[idx]
    
    return BermudanExercisePolicy(thresholds, exercise_times, K)
# EVOLVE-BLOCK-END
