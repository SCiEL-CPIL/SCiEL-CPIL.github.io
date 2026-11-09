"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction with piecewise-linear
    value function approximation.
    
    Parameters:
    -----------
    training_paths : array-like, shape (n_paths, n_time_steps)
        Simulated spot price paths.
    contract : dict
        Contract parameters.
        
    Returns:
    --------
    An object with method exercise(current_state, time) -> boolean array.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    
    # Convert training_paths to numpy array
    training_paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, total_time_steps = training_paths.shape
    
    # Determine exercise dates: 12 monthly exercise dates, no time-0 exercise
    # So exercise can happen at t=1, 2, ..., 12
    # The training paths have total_time_steps columns
    # We need to figure out the timing structure
    
    # For a Bermudan option with 12 exercise dates over T=1 year:
    # dt = T / n_steps = 1/12
    # Exercise at steps 1, 2, ..., 12 (0-indexed: indices 1 through 12)
    
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # Payoff function for put: max(K - S, 0)
    def payoff(S):
        return np.maximum(K - S, 0.0)
    
    # Backward induction to learn the exercise boundary
    # At each exercise date, we compare immediate exercise value vs continuation value
    
    # We'll use a piecewise-linear approximation for the continuation value
    # The value function V_t(S) is convex in S for a put
    
    # First, let's understand the structure of training_paths
    # If total_time_steps == n_steps + 1, then paths are at t=0, dt, 2dt, ..., T
    # Exercise dates are at t=dt, 2dt, ..., T (indices 1 through n_steps)
    
    if total_time_steps == n_steps + 1:
        # Standard structure: paths at t=0, dt, ..., T
        exercise_indices = list(range(1, n_steps + 1))
    else:
        # Try to infer: assume exercise at the last n_steps time points
        exercise_indices = list(range(total_time_steps - n_steps, total_time_steps))
    
    # For backward induction, we work backwards from the last exercise date
    # At the last exercise date (t=T), V_T(S) = payoff(S)
    # At earlier dates, V_t(S) = max(payoff(S), discount * E[V_{t+1}(S_{t+1}) | S_t = S])
    
    # Since we have simulated paths, we can use regression to estimate the continuation value
    # We'll use a simple approach: bin the state variable and compute average continuation values
    
    # Number of bins for state variable approximation
    n_bins = 50
    s_min = np.percentile(training_paths[:, exercise_indices], 1)
    s_max = np.percentile(training_paths[:, exercise_indices], 99)
    if s_max <= s_min:
        s_max = s_min + 1.0
    bin_edges = np.linspace(s_min, s_max, n_bins + 1)
    
    # Store the learned policy: for each exercise date, a threshold or lookup table
    # We'll store the continuation value function as piecewise linear
    
    # Initialize value function at final exercise date
    # V at last exercise = payoff
    # For backward induction, we need to estimate E[V_{t+1}(S_{t+1}) | S_t]
    
    # Approach: For each exercise date t, given S_t, we want to estimate the expected
    # discounted continuation value. We can do this by:
    # 1. Group paths by their S_t value (using bins)
    # 2. For each group, compute the average of discount * V_{t+1}(S_{t+1})
    
    # Start from the last exercise date and work backwards
    # V_next represents the value function at the next exercise date
    # At the final exercise date, V_next(S) = payoff(S)
    
    # We'll represent value functions as arrays: for each bin, the value
    # and we interpolate linearly between bins
    
    # Initialize with payoff at final date
    def eval_value_func(V_vals, S):
        """Evaluate piecewise-linear value function at points S."""
        # V_vals[i] is the value at bin_edges[i]
        # Use np.interp for linear interpolation
        return np.interp(S, bin_edges, V_vals)
    
    # At the final exercise date (index -1 in exercise_indices), value = payoff
    last_ex_idx = exercise_indices[-1]
    S_last = training_paths[:, last_ex_idx]
    V_next_vals = payoff(S_last)  # This is the payoff at the last exercise date
    
    # Actually, we need the value function, not just values at specific points
    # Let's restructure: for each exercise date, we'll compute the continuation value
    # by looking at what happens at the next exercise date
    
    # Better approach: 
    # For each exercise date t (from last to first):
    #   - The continuation value at S_t is discount * E[V_{t+1}(S_{t+1}) | S_t]
    #   - We estimate this by averaging V_{t+1}(S_{t+1}) over paths with similar S_t
    #   - The exercise decision: exercise if payoff(S_t) >= continuation_value(S_t)
    
    # Let's store the exercise boundary for each date
    # exercise_threshold[t] = the threshold above which we do NOT exercise (for a put,
    # we exercise when S is low)
    
    n_exercise_dates = len(exercise_indices)
    exercise_thresholds = np.zeros(n_exercise_dates)
    
    # Work backwards from the last exercise date
    # At the last exercise date, we always exercise if in the money
    # Actually, at maturity, we exercise if payoff > 0
    
    # For the last exercise date, the value is just the payoff
    # For earlier dates, we need to estimate continuation value
    
    # Let's do this more carefully with backward induction
    
    # V_at_date[i] will be the value function (as piecewise linear) at exercise date i
    # Start from the last date
    
    # At the last exercise date, V = payoff
    # We need to represent this as a function we can evaluate
    
    # For the backward step:
    # Given we're at exercise date t with state S_t,
    # the continuation value = discount * E[V_{t+1}(S_{t+1}) | S_t]
    
    # To estimate E[V_{t+1}(S_{t+1}) | S_t], we:
    # 1. For each path, compute V_{t+1}(S_{t+1}) using the known value function at t+1
    # 2. Group by S_t and take average
    
    # Let's implement this step by step
    
    # First, we need the value function at the last exercise date
    # At maturity, V_T(S) = max(K - S, 0)
    
    # We'll represent the value function at each date as a set of (S, V) pairs
    # and use interpolation
    
    # For efficiency, let's use the binned approach
    
    # Compute V at the last exercise date for all paths
    S_last = training_paths[:, last_ex_idx]
    V_last = payoff(S_last)
    
    # For the second-to-last exercise date, we need:
    # continuation_value(S_t) = discount * mean(V_last[S_{t+1}] for paths with similar S_t)
    
    # Let's do this for all dates from last to first
    
    # Store value functions as (bin_centers, values) for interpolation
    # We'll use the same bin_edges for all dates
    
    # Compute bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Value function at last exercise date
    V_func_next = np.interp(bin_centers, S_last, V_last)  # Average V for each bin
    
    # Actually, this isn't quite right. Let me reconsider.
    
    # At the last exercise date, the value IS the payoff. So V_T(S) = max(K-S, 0).
    # We can evaluate this directly.
    
    # For the second-to-last date, we need to compute:
    # For each path i: cont_val_i = discount * V_T(S_i[t+1])
    # Then, for each bin of S_t, average cont_val_i over paths in that bin
    
    # Let's implement this properly
    
    # Start with the value function at the last exercise date
    # V_last(S) = max(K - S, 0) - this is a simple function we can evaluate directly
    
    # For backward induction:
    # V_t(S) = max(payoff(S), discount * E[V_{t+1}(S_{t+1}) | S_t = S])
    
    # We'll estimate E[V_{t+1}(S_{t+1}) | S_t = S] using the training paths
    
    # Initialize: at the last exercise date, V = payoff
    # We'll store the value function at each date as an array of values at bin_centers
    
    # V_vals_next will hold the value function at the next exercise date (evaluated at bin_centers)
    # Start with the payoff function evaluated at bin_centers
    V_vals_next = payoff(bin_centers)
    
    # Now work backwards from second-to-last to first exercise date
    # For each date, compute the continuation value and determine the exercise threshold
    
    # We'll store the exercise decision boundary for each date
    # For a put, we exercise when S <= threshold
    
    # Let's compute the value function at each exercise date
    # V_vals_current will hold the value function at the current exercise date
    
    # Process exercise dates from last to first
    # exercise_indices has the column indices in training_paths for each exercise date
    
    # We need to process them in reverse order
    # For date index j from n_exercise_dates-1 down to 0:
    #   current_col = exercise_indices[j]
    #   next_col = exercise_indices[j+1] if j < n_exercise_dates-1 else None
    
    # At the last date (j = n_exercise_dates-1), V = payoff
    # At earlier dates, V = max(payoff, discount * E[V_next(S_next) | S_current])
    
    # Let's store the value functions for all dates
    all_V_vals = [None] * n_exercise_dates
    
    # Last date: V = payoff
    all_V_vals[-1] = payoff(bin_centers)
    
    # Work backwards
    for j in range(n_exercise_dates - 2, -1, -1):
        current_col = exercise_indices[j]
        next_col = exercise_indices[j + 1]
        
        S_current = training_paths[:, current_col]
        S_next = training_paths[:, next_col]
        
        # Evaluate V_next at S_next for all paths
        V_next_at_paths = np.interp(S_next, bin_centers, all_V_vals[j + 1])
        
        # Continuation value for each path
        cont_vals = discount * V_next_at_paths
        
        # For each bin of S_current, compute the average continuation value
        V_cont_at_bins = np.zeros(n_bins)
        for b in range(n_bins):
            mask = (S_current >= bin_edges[b]) & (S_current < bin_edges[b + 1])
            if np.any(mask):
                V_cont_at_bins[b] = np.mean(cont_vals[mask])
            else:
                # Interpolate from neighboring bins
                if b > 0 and b < n_bins - 1:
                    V_cont_at_bins[b] = 0.5 * (V_cont_at_bins[b-1] + V_cont_at_bins[b+1])
                elif b == 0:
                    V_cont_at_bins[b] = V_cont_at_bins[1] if n_bins > 1 else 0.0
                else:
                    V_cont_at_bins[b] = V_cont_at_bins[n_bins-2] if n_bins > 1 else 0.0
        
        # The value function at this date: V = max(payoff, continuation)
        payoff_at_bins = payoff(bin_centers)
        V_vals = np.maximum(payoff_at_bins, V_cont_at_bins)
        
        all_V_vals[j] = V_vals
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # Now determine the exercise threshold for each date
    # For a put, we exercise when S is below a threshold
    # The threshold is where payoff(S) = continuation_value(S)
    
    # For each date, find the threshold
    exercise_thresholds = np.zeros(n_exercise_dates)
    
    for j in range(n_exercise_dates):
        if all_V_vals[j] is None:
            # Use a default threshold
            exercise_thresholds[j] = K * 0.5
            continue
        
        # Find where payoff crosses the continuation value
        # payoff(S) = max(K - S, 0)
        # continuation value is all_V_vals[j] interpolated
        
        # The exercise region is where payoff >= continuation
        # For a put, this is typically S <= some threshold
        
        # Find the largest S where payoff(S) >= V_cont(S)
        # We can do this by checking each bin center
        
        exercise = np.zeros(n_bins, dtype=bool)
        for b in range(n_bins):
            s = bin_centers[b]
            p = payoff(s)
            v = all_V_vals[j][b]
            exercise[b] = (p >= v - 1e-10)
        
        # The threshold is the largest S where we exercise
        if np.any(exercise):
            # Find the transition point
            # For a put, exercise is for low S
            # Find the last bin where we exercise
            exercise_idx = np.where(exercise)[0]
            if len(exercise_idx) > 0:
                last_exercise_bin = exercise_idx[-1]
                # The threshold is between bin_edges[last_exercise_bin] and bin_edges[last_exercise_bin+1]
                if last_exercise_bin < n_bins - 1:
                    exercise_thresholds[j] = bin_edges[last_exercise_bin + 1]
                else:
                    exercise_thresholds[j] = bin_edges[-1]
            else:
                exercise_thresholds[j] = 0.0
        else:
            # Never exercise (shouldn't happen for a put at maturity)
            exercise_thresholds[j] = 0.0
    
    # For the last exercise date, we should always exercise if in the money
    # Let's verify and fix
    if n_exercise_dates > 0:
        # At maturity, exercise if K - S > 0, i.e., S < K
        exercise_thresholds[-1] = K
    
    # Create the exercise policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K, r, sigma, T, n_steps):
            self.thresholds = thresholds
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_steps = n_steps
            
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at each path.
            
            Parameters:
            -----------
            current_state : array-like, shape (n_paths,)
                Current spot prices.
            time : float or int
                Current time (or step index).
                
            Returns:
            --------
            boolean array of shape (n_paths,)
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            
            # Determine which exercise date we're at
            # time could be a step index or a time value
            # Based on the contract, exercise dates are at t = 1*dt, 2*dt, ..., 12*dt
            # where dt = T / n_steps
            
            dt = self.T / self.n_steps
            
            if isinstance(time, (int, np.integer)):
                # time is a step index
                step = int(time)
            else:
                # time is a continuous time value
                step = int(round(time / dt))
            
            # Clamp step to valid range
            step = max(1, min(step, self.n_steps))
            
            # Get the threshold for this exercise date
            # exercise_indices = [1, 2, ..., n_steps]
            # So step corresponds to exercise date index step - 1
            date_idx = step - 1
            date_idx = max(0, min(date_idx, len(self.thresholds) - 1))
            
            threshold = self.thresholds[date_idx]
            
            # Exercise if current_state <= threshold
            # For a put, we exercise when the stock price is low
            return current_state <= threshold
    
    return ExercisePolicy(exercise_thresholds, K, r, sigma, T, n_steps)
# EVOLVE-BLOCK-END
