import numpy as np
import time
from scipy.signal import savgol_filter

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Binned Backward Induction with Savitzky-Golay smoothing.
    """
    t_start = time.time()
    deadline = t_start + 100.0

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Ensure we have the right number of paths
    n_paths = training_paths.shape[0]
    
    # Exercise times: 1/12, 2/12, ..., 12/12
    # The paths array likely has columns for each time step.
    # We need to verify the shape of training_paths.
    # Usually training_paths is (n_paths, n_time_steps).
    # The problem states "12 monthly exercise dates".
    # If training_paths has 13 columns (0 to 12), we use columns 1 to 12.
    
    # Let's assume training_paths is (n_paths, n_exercise + 1) or similar.
    # We will extract the states at the exercise dates.
    
    # Identify the indices of the exercise dates in the training_paths array.
    # If the array has 13 columns (t=0 to t=12), indices 1..12 are the exercise dates.
    # If it has 12 columns, indices 0..11 might be t=1/12..12/12?
    # Standard convention: training_paths[:, i] is the state at time i/n_exercise.
    # Let's assume the columns correspond to the exercise dates directly if the count matches, 
    # or we need to interpolate. 
    # Given "12 monthly exercise dates", and typical GBM path generation, 
    # training_paths likely has shape (4096, 13) where col 0 is t=0 and cols 1-12 are t=1/12..1.
    # Or (4096, 12) where col 0 is t=1/12.
    
    # Let's check the shape.
    n_cols = training_paths.shape[1]
    
    if n_cols == n_exercise + 1:
        # Columns 1 to n_exercise are the exercise dates
        ex_indices = list(range(1, n_exercise + 1))
    elif n_cols == n_exercise:
        # Columns 0 to n_exercise-1 are the exercise dates
        ex_indices = list(range(n_exercise))
    else:
        # Fallback: assume the last n_exercise columns are the exercise dates
        ex_indices = list(range(n_cols - n_exercise, n_cols))

    # Extract states at exercise dates
    # states[k] is an array of shape (n_paths,) for the k-th exercise date
    states = [training_paths[:, idx] for idx in ex_indices]
    
    # Discount factor per step
    dt = T / n_exercise
    discount = np.exp(-r * dt)
    
    # Binning parameters
    n_bins = 2048
    # Determine bin range from the first set of states (or all states)
    # Using the range of S0 is safer, but states evolve.
    # Let's use the min/max of all states in the exercise dates to cover everything.
    all_states = np.concatenate(states)
    s_min = np.min(all_states) * 0.9
    s_max = np.max(all_states) * 1.1
    
    # Create bin edges
    bin_edges = np.linspace(s_min, s_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    # Initialize values at maturity (last exercise date)
    # The value at the last exercise date is the payoff itself, as there is no further continuation.
    # But we need to determine the boundary at the last date too.
    # Actually, at the last date, you always exercise if in the money.
    # So boundary at last date is K.
    
    # We will store the continuation values for the NEXT step for each bin.
    # Let V_bins be the value function at the current step, represented on the bins.
    
    # Start from the last step backwards.
    # At step n_exercise - 1 (the last one, index 11 if 0-indexed 0..11):
    # The value is max(K - S, 0).
    # We need to propagate this back.
    
    # Let's work backwards from the last exercise date to the first.
    # We maintain an array `values` of shape (n_paths,) which is the value of the option
    # at the *next* exercise date for each path.
    
    # Initialize values at T (after the last exercise date)
    # The last exercise date is at T. The value at T is the payoff.
    # So, for the step *before* the last exercise date, the continuation value is E[discount * payoff].
    
    # Let's iterate k from n_exercise - 1 down to 0.
    # At iteration k, we are determining the boundary for exercise date k.
    # `values` currently holds the value at date k+1 (or payoff at T if k is last).
    
    # For the last date (k = n_exercise - 1):
    # The "next value" is effectively the payoff at T.
    # But we are AT the last exercise date.
    # We compare Intrinsic vs Continuation.
    # Continuation at the last date is 0 (no future steps).
    # So we always exercise if Intrinsic > 0.
    # Boundary is K.
    
    # Let's set up the loop properly.
    # `next_values` = values at t_{k+1}
    
    # Initialize next_values as the payoff at T
    # The states at the last date are states[-1]
    # But wait, the payoff is realized at T.
    # If we are at t_{n-1} (last exercise), the continuation is 0.
    # So we exercise if S < K.
    
    # Let's start with k = n_exercise - 1.
    # next_values is not really used for continuation at the last step, it's 0.
    # But for the step before that, we need the values.
    
    # Let's refine the loop:
    # We want to find boundaries[0] ... boundaries[n_exercise-1].
    
    boundaries = [0.0] * n_exercise
    
    # Step 1: Last exercise date (index n_exercise - 1)
    # Boundary is K.
    boundaries[-1] = K
    
    # Value at the last exercise date for each path:
    # V_last = max(K - S_last, 0)
    S_last = states[-1]
    V_current = np.maximum(K - S_last, 0.0)
    
    # Now we go back to the second to last date (index n_exercise - 2)
    # We need the continuation value at t_{n-2} which is discount * E[V_last | S_{n-2}]
    
    for k in range(n_exercise - 2, -1, -1):
        if time.time() > deadline:
            break
            
        S_curr = states[k]
        
        # 1. Bin the current states
        # Map each path's state to a bin index
        # bin_idx = floor((S - s_min) / bin_width)
        # Clip to [0, n_bins-1]
        bin_width = (s_max - s_min) / n_bins
        bin_indices = np.floor((S_curr - s_min) / bin_width).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # 2. Calculate the mean of next_values for each bin
        # We use a trick with np.bincount or similar.
        # We want sum(next_values * discount) for each bin, then divide by count.
        # Actually, we want the expected value.
        
        # Sum of discounted next values per bin
        weighted_values = discount * V_current
        sum_per_bin = np.bincount(bin_indices, weights=weighted_values, minlength=n_bins)
        count_per_bin = np.bincount(bin_indices, minlength=n_bins).astype(float)
        
        # Avoid division by zero
        count_per_bin[count_per_bin == 0] = 1.0
        mean_cont_per_bin = sum_per_bin / count_per_bin
        
        # 3. Smooth the continuation values
        # Only smooth if we have enough data points in the bins
        # Savitzky-Golay requires a window size.
        # If n_bins is 2048, window 11 is fine.
        try:
            smoothed_cont = savgol_filter(mean_cont_per_bin, window_length=11, polyorder=3)
        except Exception:
            smoothed_cont = mean_cont_per_bin
            
        # 4. Determine exercise boundary
        # Intrinsic value at bin centers
        intrinsic_per_bin = np.maximum(K - bin_centers, 0.0)
        
        # Exercise if Intrinsic > Smoothed Continuation
        exercise_mask = intrinsic_per_bin > smoothed_cont
        
        # Find the maximum bin index where exercise is optimal
        # The boundary is the state value corresponding to this bin.
        if np.any(exercise_mask):
            # We want the highest S where we still exercise.
            # So we find the max index in exercise_mask.
            max_ex_idx = np.max(np.where(exercise_mask)[0])
            # The boundary is the center of that bin (or the upper edge?)
            # Using the center is a good approximation.
            boundaries[k] = bin_centers[max_ex_idx]
        else:
            boundaries[k] = 0.0
            
        # 5. Update V_current for the next iteration (which will be the previous step)
        # For each path, the value is max(Intrinsic, Continuation)
        # We need to interpolate the smoothed continuation value for each path's state.
        
        # Interpolate smoothed_cont to the current states
        # np.interp requires the x-coordinates to be increasing. bin_centers are increasing.
        cont_vals_interp = np.interp(S_curr, bin_centers, smoothed_cont)
        
        # Intrinsic value for each path
        intrinsic_vals = np.maximum(K - S_curr, 0.0)
        
        # Value at this step
        V_current = np.maximum(intrinsic_vals, cont_vals_interp)
        
    # Ensure monotonicity: boundaries[k] <= boundaries[k+1]
    # For a put, earlier dates have lower boundaries.
    for k in range(n_exercise - 1):
        if boundaries[k] > boundaries[k + 1]:
            boundaries[k] = boundaries[k + 1]
            
    # Cap boundaries at K
    for k in range(n_exercise):
        if boundaries[k] > K:
            boundaries[k] = K

    class ExercisePolicy:
        def __init__(self, K, boundaries, n_exercise, T):
            self.K = K
            self.boundaries = boundaries
            self.n_exercise = n_exercise
            self.T = T
        
        def exercise(self, current_state, time):
            n_paths = len(current_state)
            result = np.zeros(n_paths, dtype=bool)
            
            # Determine which exercise date we're at based on time
            # time is in [0, 1]. Exercise dates are at i/n_exercise.
            # We need to find the closest exercise date index.
            # i = round(time * n_exercise)
            # If i == 0, no exercise (t=0 is not an exercise date in this contract? 
            # "12 monthly exercise dates (no time-0 exercise)").
            # So t=1/12, ..., 12/12.
            
            idx_float = time * self.n_exercise
            idx = int(round(idx_float))
            
            # Clamp idx to [1, n_exercise]
            if idx < 1:
                idx = 1
            elif idx > self.n_exercise:
                idx = self.n_exercise
                
            # boundaries is 0-indexed.
            # boundaries[0] corresponds to t=1/12.
            # boundaries[11] corresponds to t=12/12.
            # So we use boundaries[idx - 1].
            
            threshold = self.boundaries[idx - 1]
            
            # Exercise if S <= threshold
            result = (current_state <= threshold)
            
            return result

    policy = ExercisePolicy(K, boundaries, n_exercise, T)
    return policy
