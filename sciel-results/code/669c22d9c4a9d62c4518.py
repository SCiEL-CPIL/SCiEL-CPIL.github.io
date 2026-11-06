"""Bermudan exercise-policy research: fit optimal exercise thresholds via backward induction on training paths."""
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit an exercise policy for a Bermudan put.
    
    Args:
        training_paths: array of shape (n_paths, n_time_points) with GBM paths
        contract: dict with keys 'K', 'r', 'sigma', 'T', 'n_exercise_dates', 'S0', etc.
    
    Returns:
        An object with method exercise(current_state, time) -> boolean vector
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise_dates', 12)
    S0 = contract.get('S0', 100.0)
    
    # Determine number of time points from training paths
    # training_paths shape: (n_paths, n_time_points)
    # For Bermudan with 12 exercise dates (no time-0), we have times t_1, ..., t_12
    # So training_paths should have 13 columns (t_0 to t_12) or 12 columns (t_1 to t_12)
    if training_paths is None or len(training_paths) == 0:
        # Fallback: create a simple threshold policy
        return _SimpleThresholdPolicy(K, r, sigma, T, n_exercise)
    
    n_paths, n_time_pts = training_paths.shape
    
    # The exercise dates are at t_1, ..., t_12 (no exercise at t_0)
    # training_paths likely contains values at all time points including t_0
    # We need to figure out the mapping
    # Typically for Bermudan, we have n_exercise exercise dates
    # If n_time_pts == n_exercise + 1, then columns are t_0, t_1, ..., t_n_exercise
    # If n_time_pts == n_exercise, then columns are t_1, ..., t_n_exercise
    
    if n_time_pts == n_exercise + 1:
        # Columns 0..n_exercise correspond to t_0..t_n_exercise
        # Exercise can happen at indices 1..n_exercise (times t_1..t_n_exercise)
        exercise_indices = list(range(1, n_exercise + 1))
    elif n_time_pts == n_exercise:
        # Columns 0..n_exercise-1 correspond to t_1..t_n_exercise
        exercise_indices = list(range(0, n_exercise))
    else:
        # Fallback
        return _SimpleThresholdPolicy(K, r, sigma, T, n_exercise)
    
    dt = T / n_exercise
    discount = np.exp(-r * dt)
    
    # We'll use backward induction on the training paths to find optimal exercise thresholds
    # For each exercise date, we want to find the threshold price below which it's optimal to exercise
    
    # The value at the last exercise date (t_12) is simply the payoff if in-the-money
    # For earlier dates, we compare immediate exercise payoff vs. continuation value
    
    # Strategy: For each exercise date i (from last to first), determine the optimal threshold
    # by analyzing the training paths
    
    # Let's define:
    # V[i] = optimal value at exercise date i given current price
    # At the last exercise date (index -1 in exercise_indices), V = max(K - S, 0)
    # For earlier dates: V = max(K - S, discount * E[V_next | S])
    
    # Since we're working with discrete paths, we'll estimate the continuation value
    # by looking at the average of next-step values for paths at similar current prices
    
    # For efficiency, we'll bin the price levels and compute average continuation values
    
    # Number of exercise dates
    n_ex = len(exercise_indices)
    
    # Initialize thresholds array
    thresholds = np.zeros(n_ex)
    
    # Start from the last exercise date
    # At the last date, exercise if K - S > 0, i.e., S < K
    thresholds[-1] = K
    
    # Work backwards
    for i in range(n_ex - 2, -1, -1):
        if time.time() > deadline - 5.0:
            # Not enough time, use simple heuristic for remaining
            for j in range(i, -1, -1):
                thresholds[j] = K * (1 - 0.02 * (j + 1))
            break
        
        idx_i = exercise_indices[i]
        idx_next = exercise_indices[i + 1]
        
        # Current prices at exercise date i
        S_i = training_paths[:, idx_i]
        # Next prices at exercise date i+1
        S_next = training_paths[:, idx_next]
        
        # Continuation value at next date: apply the threshold for date i+1
        # If S_next < thresholds[i+1], exercise: payoff = K - S_next
        # Otherwise, continue: but we need the value, which for the last date is just payoff
        # For intermediate dates, we need to propagate values
        
        # Actually, we need to compute the full continuation value, not just the threshold
        # Let's recompute values properly
        
        # For the last exercise date, value = max(K - S, 0)
        # For earlier dates, value = max(K - S, discount * continuation_value)
        
        # Let's compute values backwards
        # V_next = value at the next exercise date for each path
        # At the last date: V = max(K - S, 0)
        
        # Recompute: we need the actual values, not just thresholds
        # Let's redo this more carefully
        
        # V at last exercise date
        V_next = np.maximum(K - S_next, 0.0)
        
        # But wait, for the last date, if we don't exercise, the option expires worthless
        # So V at last date = max(K - S, 0) is correct
        
        # For the second-to-last date:
        # Continuation value = discount * V_next (since V_next already accounts for optimal exercise at next date)
        # Exercise value = max(K - S_i, 0)
        # Optimal: exercise if exercise_value > continuation_value
        
        # However, V_next depends on the optimal policy at the next date, which we've determined
        # by the threshold. So we need to compute V_next using the threshold for date i+1.
        
        # Let me restructure: compute values from last to first
        
        # Restart the backward induction properly
        break
    
    # Let's do a clean backward induction
    # Compute value arrays from last to first exercise date
    
    # V at last exercise date
    idx_last = exercise_indices[-1]
    S_last = training_paths[:, idx_last]
    V = np.maximum(K - S_last, 0.0)
    
    # Threshold at last date
    # Exercise if K - S > 0, so threshold = K
    thresholds[-1] = K
    
    # Work backwards from second-to-last to first
    for i in range(n_ex - 2, -1, -1):
        if time.time() > deadline - 5.0:
            # Use heuristic for remaining
            for j in range(i, -1, -1):
                # Estimate threshold based on time to maturity
                # Earlier dates have lower thresholds (less time value)
                t_remaining = (i + 1) * dt
                # Simple approximation: threshold decreases as time to maturity decreases
                # For a put, the critical price is lower when there's less time
                thresholds[j] = K * np.exp(-0.5 * sigma * np.sqrt(t_remaining))
            break
        
        idx_i = exercise_indices[i]
        idx_next = exercise_indices[i + 1]
        
        S_i = training_paths[:, idx_i]
        S_next = training_paths[:, idx_next]
        
        # V currently holds the value at exercise date i+1 (already computed)
        # Continuation value at date i = discount * V
        cont_value = discount * V
        
        # Exercise value at date i
        ex_value = np.maximum(K - S_i, 0.0)
        
        # Determine optimal exercise: exercise if ex_value > cont_value
        exercise_mask = ex_value > cont_value
        
        # The value at date i
        V_i = np.where(exercise_mask, ex_value, cont_value)
        
        # Estimate the threshold for date i
        # The threshold is the price level where ex_value = cont_value
        # For simplicity, find the median price of paths that exercise vs. don't
        if np.any(exercise_mask) and np.any(~exercise_mask):
            # Threshold is approximately the median of the boundary
            S_ex = S_i[exercise_mask]
            S_no_ex = S_i[~exercise_mask]
            # The threshold is between max(S_ex) and min(S_no_ex)
            if len(S_ex) > 0 and len(S_no_ex) > 0:
                thresholds[i] = (np.max(S_ex) + np.min(S_no_ex)) / 2.0
            else:
                thresholds[i] = K
        elif np.all(exercise_mask):
            # All paths exercise, threshold is high
            thresholds[i] = np.max(S_i) if len(S_i) > 0 else K
        else:
            # No paths exercise, threshold is low
            thresholds[i] = np.min(S_i) if len(S_i) > 0 else 0.0
        
        # Update V for next iteration
        V = V_i
    
    # Now we have thresholds for each exercise date
    # Create the policy object
    policy = _BermudanPolicy(thresholds, n_exercise, K, r, sigma, T, n_ex)
    
    return policy


class _BermudanPolicy:
    """Exercise policy based on precomputed thresholds for each exercise date."""
    
    def __init__(self, thresholds, n_exercise, K, r, sigma, T, n_ex):
        self.thresholds = thresholds
        self.n_exercise = n_exercise
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_ex = n_ex
    
    def exercise(self, current_state, time):
        """
        Determine whether to exercise the Bermudan put.
        
        Args:
            current_state: array of current spot prices, shape (n_paths,)
            time: current time (scalar or array)
        
        Returns:
            Boolean array indicating whether to exercise at each path
        """
        # Determine which exercise date we're at
        # time is in [0, T], exercise dates are at t_1, ..., t_n_exercise
        # t_i = i * T / n_exercise for i = 1, ..., n_exercise
        
        if isinstance(time, (int, float, np.integer, np.floating)):
            # Scalar time
            # Find the exercise date index
            dt = self.T / self.n_exercise
            # Exercise dates are at dt, 2*dt, ..., n_exercise*dt
            # time should be close to one of these
            idx = int(round(time / dt))
            # Clamp to valid range
            idx = max(1, min(idx, self.n_exercise))
            # Map to threshold index: exercise date idx corresponds to threshold index idx-1
            threshold_idx = idx - 1
            threshold = self.thresholds[threshold_idx]
            return current_state < threshold
        else:
            # Array of times
            times = np.asarray(time)
            dt = self.T / self.n_exercise
            idxs = np.round(times / dt).astype(int)
            idxs = np.clip(idxs, 1, self.n_exercise)
            threshold_idxs = idxs - 1
            thresholds = self.thresholds[threshold_idxs]
            return current_state < thresholds


class _SimpleThresholdPolicy:
    """Fallback simple threshold policy."""
    
    def __init__(self, K, r, sigma, T, n_exercise):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_exercise = n_exercise
        # Precompute thresholds using a simple model
        dt = T / n_exercise
        self.thresholds = []
        for i in range(1, n_exercise + 1):
            t_remaining = (n_exercise - i) * dt
            # Simple approximation for Bermudan put threshold
            # The threshold is below K, and decreases as time to maturity decreases
            # Use a rough approximation
            if t_remaining <= 0:
                self.thresholds.append(K)
            else:
                # Approximate using the idea that the critical price for a put
                # is roughly K * exp(-sigma * sqrt(t_remaining) / 2) for small sigma
                # This is a rough heuristic
                threshold = K * np.exp(-0.3 * sigma * np.sqrt(t_remaining))
                self.thresholds.append(max(threshold, 0.0))
    
    def exercise(self, current_state, time):
        if isinstance(time, (int, float, np.integer, np.floating)):
            dt = self.T / self.n_exercise
            idx = int(round(time / dt))
            idx = max(1, min(idx, self.n_exercise))
            threshold_idx = idx - 1
            return current_state < self.thresholds[threshold_idx]
        else:
            times = np.asarray(time)
            dt = self.T / self.n_exercise
            idxs = np.round(times / dt).astype(int)
            idxs = np.clip(idxs, 1, self.n_exercise)
            threshold_idxs = idxs - 1
            thresholds = np.array(self.thresholds)[threshold_idxs]
            return current_state < thresholds
