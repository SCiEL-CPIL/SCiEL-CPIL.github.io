"""Rewritten program using direct Monte Carlo optimization of exercise thresholds."""
# EVOLVE-BLOCK-START
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using stochastic gradient ascent on
    12 threshold parameters (one per exercise date).
    
    The policy at each exercise date t_i is: exercise if S_t < threshold_i.
    We optimize thresholds to maximize expected discounted payoff.
    """
    # Parse contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Exercise dates: 12 monthly dates, no time-0
    # T=1 year, 12 monthly dates => at t = 1/12, 2/12, ..., 12/12
    exercise_times = np.linspace(1.0/12.0, 1.0, n_exercise)
    
    # training_paths shape: (n_paths, n_time_steps) or similar
    # Need to figure out the structure. Let's assume it's a 2D array where
    # each row is a path and columns are time steps including t=0.
    # With 12 monthly exercise dates over T=1, and if paths are generated
    # at monthly intervals, we'd have 13 time points (t=0 to t=1).
    
    n_paths = training_paths.shape[0]
    n_time_steps = training_paths.shape[1]
    
    # Map exercise dates to column indices
    # If n_time_steps = 13 (t=0, 1/12, ..., 12/12), then exercise date i (1-indexed)
    # is at column i.
    # In general, we need to find which column corresponds to each exercise time.
    # Assume uniform spacing: dt = T / (n_time_steps - 1)
    if n_time_steps > 1:
        dt = T / (n_time_steps - 1)
    else:
        dt = T
    
    # For each exercise time, find the nearest column index
    exercise_cols = []
    for t_ex in exercise_times:
        col = int(round(t_ex / dt))
        col = min(col, n_time_steps - 1)
        exercise_cols.append(col)
    
    # Initialize thresholds - start with K (strike price) as a reasonable initial guess
    thresholds = np.full(n_exercise, K)
    
    # Stochastic gradient ascent
    n_iter = 200
    lr = 0.5  # learning rate for threshold updates
    
    # We'll use mini-batches for stochastic gradient
    batch_size = min(512, n_paths)
    
    for it in range(n_iter):
        # Shuffle and take a batch
        indices = np.random.choice(n_paths, batch_size, replace=False)
        batch_paths = training_paths[indices]
        
        # Compute current policy value and gradient estimate
        # For each exercise date from last to first (backward), we can compute
        # the marginal benefit of exercising vs continuing.
        
        # Actually, for a threshold policy, the gradient with respect to threshold_i
        # can be estimated by:
        # dV/dtheta_i = E[ (payoff_if_exercise - continuation_value) * indicator(S near theta_i) * sign ]
        # 
        # A simpler approach: use finite differences or a direct heuristic.
        # Let's use a simple approach: for each threshold, estimate the gradient
        # by comparing paths that exercise vs don't.
        
        # Compute continuation values via backward induction on the batch
        # Discount factors
        disc = np.exp(-r * np.array([exercise_times[i+1] - exercise_times[i] if i < n_exercise-1 else 0 for i in range(n_exercise)]))
        # Actually, discount from each exercise date to the next
        # disc[i] = exp(-r * (t_{i+1} - t_i)) for i < n-1, and for the last one, no discount needed
        # But we need to be careful: the payoff at exercise date i is received at t_i.
        # When comparing to continuation, we discount the future value back to t_i.
        
        # Let's recompute: for exercise date i, the continuation value is the expected
        # value of waiting, which involves discounting from t_{i+1} to t_i.
        # disc_to_next[i] = exp(-r * (t_{i+1} - t_i)) for i < n-1
        # For the last exercise date, there's no continuation (or we can consider it as 0 if we must exercise or not).
        
        # Actually, for a Bermudan option, at the last exercise date, you either exercise or not.
        # If you don't exercise, the option expires worthless (for a put, if S > K, it's worthless).
        # So continuation value at the last date is 0 if you choose not to exercise.
        
        # Let's do backward induction on the batch to compute continuation values.
        # cont_value[i, j] = continuation value at exercise date i for path j
        # Start from the last exercise date.
        
        cont_values = np.zeros((n_exercise, batch_size))
        
        # Last exercise date: continuation value is 0 (if you don't exercise, option expires)
        # cont_values[-1] = 0
        
        # For i from n-2 down to 0:
        # cont_values[i, j] = disc_to_next[i] * E[ max(payoff at i+1, cont_values[i+1]) | S_i ]
        # But we don't have conditional expectations easily. Instead, we can approximate.
        
        # Alternative: compute the full value via backward induction assuming the current thresholds,
        # then estimate gradients.
        
        # Let's compute the policy value and use a simple gradient estimate.
        # For each threshold_i, the gradient is proportional to:
        # E[ (immediate_payoff - disc * future_value) * d(indicator)/dtheta ]
        # d(indicator(S < theta))/dtheta = delta(S - theta) (Dirac delta)
        # In practice, we can use a finite difference:
        # gradient_i ≈ (value_with_threshold_i + eps - value_with_threshold_i - eps) / (2*eps)
        
        # This is expensive. Let's use a simpler heuristic:
        # For each exercise date, count how many paths have S near the threshold,
        # and adjust the threshold in the direction that increases the average payoff.
        
        # Simpler approach: use the fact that the optimal threshold for a put is below K.
        # Use a local search: for each threshold, try a small perturbation and see if it improves.
        
        # Let's use a different approach: simulate the policy on the batch and compute
        # the average payoff. Then use a simple gradient-free method like Nelder-Mead
        # or just coordinate descent with finite differences.
        
        # Given time constraints, let's use coordinate descent with finite differences.
        # For each threshold, try +eps and -eps, and keep the better one.
        
        eps = 0.1
        
        def compute_policy_value(threshs, paths):
            """Compute the expected discounted payoff for a given set of thresholds."""
            n_p = paths.shape[0]
            total_value = 0.0
            
            # Process from last exercise date to first
            # We need to track which paths are still alive (not yet exercised)
            alive = np.ones(n_p, dtype=bool)
            
            # We'll store the continuation values
            # For each path, we track the value if we continue
            # This is complex. Let's do a forward simulation with backward-looking decisions.
            
            # Actually, let's do backward induction properly.
            # At each exercise date, for each path, we decide whether to exercise.
            # The decision depends on the continuation value, which depends on future decisions.
            
            # Let's precompute the continuation values via backward induction.
            # cont[i, j] = value of continuing at exercise date i for path j
            # For the last exercise date: cont[n-1, j] = 0 (if we don't exercise, option expires)
            # For i < n-1: cont[i, j] = disc[i] * (payoff_if_exercise_at_i+1 if S_{i+1} < thresh[i+1] else cont[i+1, j])
            # But this assumes we know the future states, which we do in the training paths.
            
            # This is a Monte Carlo estimate of the continuation value.
            
            # Initialize cont for the last date
            cont = np.zeros((n_exercise, n_p))
            # cont[-1, :] = 0
            
            # Backward induction
            for i in range(n_exercise - 2, -1, -1):
                # Discount factor from t_{i+1} to t_i
                dt_next = exercise_times[i+1] - exercise_times[i]
                disc_i = np.exp(-r * dt_next)
                
                # At exercise date i+1, for each path, the value is:
                # max(payoff_if_exercise, cont[i+1, j])
                # payoff_if_exercise = max(K - S_{i+1}, 0)
                S_next = paths[:, exercise_cols[i+1]]
                payoff_next = np.maximum(K - S_next, 0.0)
                
                # Decision at i+1: exercise if S < thresh[i+1]
                exercise_next = S_next < threshs[i+1]
                
                # Value at i+1 given the policy
                value_next = np.where(exercise_next, payoff_next, cont[i+1, :])
                
                # Discount back to i
                cont[i, :] = disc_i * value_next
            
            # Now compute the total value at time 0 (or first exercise date)
            # At the first exercise date, the value is:
            # max(payoff_if_exercise, cont[0, j])
            S_first = paths[:, exercise_cols[0]]
            payoff_first = np.maximum(K - S_first, 0.0)
            exercise_first = S_first < threshs[0]
            
            value_first = np.where(exercise_first, payoff_first, cont[0, :])
            
            # The option value at time 0 is the discounted value at the first exercise date
            dt_first = exercise_times[0]
            disc_first = np.exp(-r * dt_first)
            
            total_value = disc_first * np.mean(value_first)
            
            return total_value
        
        # Coordinate descent with finite differences
        improved = False
        for i in range(n_exercise):
            # Try +eps
            thresh_plus = thresholds.copy()
            thresh_plus[i] += eps
            val_plus = compute_policy_value(thresh_plus, batch_paths)
            
            # Try -eps
            thresh_minus = thresholds.copy()
            thresh_minus[i] -= eps
            val_minus = compute_policy_value(thresh_minus, batch_paths)
            
            # Current value
            val_curr = compute_policy_value(thresholds, batch_paths)
            
            # Choose the best
            if val_plus > val_curr and val_plus >= val_minus:
                thresholds[i] += eps
                improved = True
            elif val_minus > val_curr and val_minus >= val_plus:
                thresholds[i] -= eps
                improved = True
        
        # Optional: stop if no improvement
        if not improved:
            break
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, exercise_cols, K):
            self.thresholds = thresholds
            self.exercise_cols = exercise_cols
            self.K = K
            self.n_exercise = len(thresholds)
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            
            Parameters:
            - current_state: array of current spot prices (one per path)
            - time: current time
            
            Returns:
            - boolean vector indicating whether to exercise for each path
            """
            # Find the exercise date index corresponding to the current time
            # time is a scalar or array? Assume scalar for the exercise date.
            # We need to map time to the nearest exercise date index.
            
            # Find the index of the exercise date closest to time
            if isinstance(time, (int, float, np.floating)):
                t_val = float(time)
                # Find the exercise date index
                idx = np.argmin(np.abs(exercise_times - t_val))
                # Check if it's close enough
                if abs(exercise_times[idx] - t_val) > 0.01:
                    # Not an exercise date, return all False
                    return np.zeros(len(current_state), dtype=bool)
                
                # Exercise if S < threshold
                return current_state < self.thresholds[idx]
            else:
                # If time is an array, handle each element
                # This is unlikely, but let's handle it
                result = np.zeros(len(current_state), dtype=bool)
                # Assume time is a scalar for each path? Or a single time for all paths?
                # The contract says "returning a boolean vector", so time is likely a scalar.
                # If time is an array, we need to handle it differently.
                # For now, assume scalar.
                t_val = float(time[0]) if hasattr(time, '__len__') else float(time)
                idx = np.argmin(np.abs(exercise_times - t_val))
                if abs(exercise_times[idx] - t_val) > 0.01:
                    return np.zeros(len(current_state), dtype=bool)
                return current_state < self.thresholds[idx]
    
    return ExercisePolicy(thresholds, exercise_cols, K)
# EVOLVE-BLOCK-END
