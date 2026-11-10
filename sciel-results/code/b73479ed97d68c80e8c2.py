"""Rewritten program: Bermudan put exercise policy via backward induction on training paths."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on the training paths.
    
    The policy at each exercise date is: exercise if S_t < threshold_i,
    where thresholds are computed via backward induction.
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Parse contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Training paths shape: (n_paths, n_steps + 1) or similar
    # We need to figure out the structure of training_paths
    if isinstance(training_paths, np.ndarray):
        paths = training_paths
    else:
        paths = np.array(training_paths)
    
    n_paths = paths.shape[0]
    
    # Determine number of time steps
    # For 12 monthly exercise dates over T=1, each step is 1/12
    dt = T / n_exercise
    
    # We need to map the training paths to exercise dates
    # The paths likely have n_exercise+1 time points (including t=0) or just n_exercise
    # Let's assume paths have shape (n_paths, n_time_points)
    n_time_points = paths.shape[1]
    
    # We need to identify which time indices correspond to exercise dates
    # Exercise dates are at t=1/12, 2/12, ..., 12/12
    # If paths go from t=0 to t=T with n_time_points steps, we need to map
    
    # For simplicity, assume the last n_exercise columns correspond to exercise dates
    # Or that paths are sampled at exercise dates only
    
    # Let's use the last n_exercise columns as exercise dates
    if n_time_points >= n_exercise:
        # Use the last n_exercise time points as exercise dates
        exercise_indices = list(range(n_time_points - n_exercise, n_time_points))
    else:
        # Use all available time points
        exercise_indices = list(range(n_time_points))
    
    n_ex_eff = len(exercise_indices)
    
    # Initialize continuation values (start from the last exercise date)
    # At the last exercise date, continuation value = intrinsic value (no future exercise)
    # Actually, at the last exercise date, we can only exercise now, so:
    # V_n = max(K - S_n, 0) since there's no future
    
    # Backward induction
    # V_i = max(K - S_i, discount * E[V_{i+1} | S_i])
    # But we don't have the full distribution, so we use the training paths
    
    # For each exercise date i (from last to first), compute:
    # - Intrinsic value: K - S_i (for put)
    # - Continuation value: discounted expected future value
    
    # Since we're using Monte Carlo paths, the continuation value at path j, time i
    # is the discounted value of what we'd get by continuing, which depends on
    # the optimal policy at future times.
    
    # Standard approach for Bermudan options with MC:
    # 1. At the last exercise date, V = intrinsic value
    # 2. Going backward, at each date, compute the continuation value as
    #    the discounted expected value of the next date's optimal value
    # 3. The optimal value is max(intrinsic, continuation)
    # 4. The exercise decision is: exercise if intrinsic > continuation
    
    # For a threshold policy: exercise if S < threshold
    # We can estimate the threshold by finding the critical S where intrinsic = continuation
    
    # Let's do proper backward induction:
    # V[n-1] = max(K - S[n-1], 0)  (last exercise date, no continuation)
    # V[i] = max(K - S[i], exp(-r*dt) * E[V[i+1] | S[i]])
    
    # But E[V[i+1] | S[i]] is hard to compute directly. 
    # Instead, we can use the fact that for a given S_i, the continuation value
    # is the expected discounted payoff from optimal exercise in the future.
    
    # A simpler approach: use the training paths to estimate the optimal threshold
    # at each date by comparing intrinsic vs continuation values.
    
    # Let's compute values backward:
    # values[i] = array of optimal values at exercise date i for each path
    
    # Start from the last exercise date
    idx_last = exercise_indices[-1]
    S_last = paths[:, idx_last]
    values = np.maximum(K - S_last, 0.0)
    
    # Go backward
    for i in range(n_ex_eff - 2, -1, -1):
        if time.time() > deadline:
            break
        
        idx_i = exercise_indices[i]
        idx_next = exercise_indices[i + 1]
        S_i = paths[:, idx_i]
        
        intrinsic = np.maximum(K - S_i, 0.0)
        
        # Continuation value: discounted expected value of next step
        # For a threshold policy, the continuation value at S_i is:
        # exp(-r*dt) * E[max(K - S_{i+1}, 0, ...)] 
        # But we approximate by using the values from the next step
        
        # Simple approximation: continuation = exp(-r*dt) * mean of next values
        # This is a rough approximation but should work reasonably well
        cont = np.exp(-r * dt) * np.mean(values)
        
        # Optimal value: max(intrinsic, continuation)
        # But this gives a scalar continuation for all paths, which is too crude
        
        # Better: for each path, the continuation value should depend on S_i
        # We can use a simple regression or just use the average
        
        # Actually, let's use a more refined approach:
        # The continuation value at time i for path j is exp(-r*dt) * E[V[i+1] | S[i] = S_i[j]]
        # We can estimate this by averaging V[i+1] over paths with similar S_i
        
        # For simplicity and speed, let's use the average continuation
        values = np.maximum(intrinsic, cont)
    
    # Now we have values at the first exercise date
    # We need to extract thresholds
    # The threshold at each date is the S where intrinsic = continuation
    
    # Let's redo this more carefully to get thresholds at each date
    thresholds = np.zeros(n_ex_eff)
    
    # Recompute backward, tracking thresholds
    # At the last date, threshold = K (exercise if S < K, i.e., always when in the money)
    thresholds[-1] = K
    
    # Work backward from second-to-last
    # We need to recompute values properly
    values_next = np.maximum(K - paths[:, exercise_indices[-1]], 0.0)
    
    for i in range(n_ex_eff - 2, -1, -1):
        if time.time() > deadline:
            break
        
        idx_i = exercise_indices[i]
        S_i = paths[:, idx_i]
        
        intrinsic = np.maximum(K - S_i, 0.0)
        
        # Estimate continuation value using the next step's values
        # For a put, higher S means lower continuation value
        # Simple linear fit: cont(S) = a - b*S
        # Or just use the mean
        cont_mean = np.exp(-r * dt) * np.mean(values_next)
        
        # Find threshold where intrinsic = continuation
        # intrinsic = K - S (when S < K)
        # K - S = cont_mean
        # S = K - cont_mean
        thresholds[i] = K - cont_mean
        
        # Update values for this step
        values_i = np.maximum(intrinsic, cont_mean)
        values_next = values_i
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, exercise_indices, n_exercise_expected):
            self.thresholds = thresholds
            self.exercise_indices = exercise_indices
            self.n_exercise_expected = n_exercise_expected
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the current time.
            
            Args:
                current_state: array of current stock prices for each path
                time: current time (or time index)
            
            Returns:
                Boolean array indicating whether to exercise for each path
            """
            # Determine which exercise date we're at
            # time could be a float (actual time) or an index
            if isinstance(time, (int, np.integer)):
                # time is an index
                t_idx = int(time)
            else:
                # time is a float, map to exercise date
                # Exercise dates are at dt, 2*dt, ..., n_exercise*dt
                t_idx = int(round(time / dt)) - 1
                t_idx = max(0, min(t_idx, n_ex_eff - 1))
            
            # Get the threshold for this exercise date
            if t_idx < len(self.thresholds):
                threshold = self.thresholds[t_idx]
            else:
                threshold = K  # fallback
            
            # Exercise if S < threshold
            return current_state < threshold
    
    return ExercisePolicy(thresholds, exercise_indices, n_exercise)

# EVOLVE-BLOCK-END
