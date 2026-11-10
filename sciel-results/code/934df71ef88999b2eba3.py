"""
Bermudan put exercise policy via Monte Carlo threshold optimization.
Uses backward induction on training paths to compute optimal thresholds
through coordinate descent with full-batch evaluations.
"""
# EVOLVE-BLOCK-START
import numpy as np


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy by optimizing 12 threshold parameters
    (one per exercise date) to maximize expected discounted payoff.
    
    Policy: at exercise date i, exercise if S_t < threshold_i.
    Optimization: coordinate descent with finite differences on full training batch.
    """
    # Parse contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Exercise dates: 12 monthly dates, no time-0
    exercise_times = np.linspace(1.0/12.0, 1.0, n_exercise)
    
    n_paths = training_paths.shape[0]
    n_time_steps = training_paths.shape[1]
    
    # Map exercise dates to column indices
    if n_time_steps > 1:
        dt = T / (n_time_steps - 1)
    else:
        dt = T
    
    exercise_cols = []
    for t_ex in exercise_times:
        col = int(round(t_ex / dt))
        col = min(col, n_time_steps - 1)
        exercise_cols.append(col)
    
    # Extract spot prices at each exercise date
    S_at_exercise = np.column_stack([training_paths[:, col] for col in exercise_cols])
    # Shape: (n_paths, n_exercise)
    
    # Precompute discount factors between consecutive exercise dates
    # disc[i] = exp(-r * (t_{i+1} - t_i)) for i = 0..n-2
    # For the last date, no continuation
    disc_factors = np.ones(n_exercise)
    for i in range(n_exercise - 1):
        dt_step = exercise_times[i+1] - exercise_times[i]
        disc_factors[i] = np.exp(-r * dt_step)
    
    # Discount from time 0 to first exercise date
    disc_to_first = np.exp(-r * exercise_times[0])
    
    # Initialize thresholds - start with K as a reasonable initial guess
    # For a put, optimal thresholds are typically below K
    thresholds = np.full(n_exercise, K)
    
    def compute_policy_value(threshs):
        """
        Compute expected discounted payoff for given thresholds using
        backward induction on all training paths.
        """
        n_p = n_paths
        # cont[i, j] = continuation value at exercise date i for path j
        # For the last exercise date: if we don't exercise, option expires -> 0
        cont = np.zeros((n_exercise, n_p))
        
        # Backward induction from last to first
        for i in range(n_exercise - 2, -1, -1):
            S_next = S_at_exercise[:, i+1]
            payoff_next = np.maximum(K - S_next, 0.0)
            
            # Decision at i+1: exercise if S < threshs[i+1]
            exercise_next = S_next < threshs[i+1]
            
            # Value at i+1 given the policy
            value_next = np.where(exercise_next, payoff_next, cont[i+1, :])
            
            # Discount back to i
            cont[i, :] = disc_factors[i] * value_next
        
        # At the first exercise date
        S_first = S_at_exercise[:, 0]
        payoff_first = np.maximum(K - S_first, 0.0)
        exercise_first = S_first < threshs[0]
        
        value_first = np.where(exercise_first, payoff_first, cont[0, :])
        
        # Discount to time 0
        total_value = disc_to_first * np.mean(value_first)
        
        return total_value
    
    # Coordinate descent with adaptive step size
    n_iter = 30
    base_eps = 0.5
    
    for it in range(n_iter):
        improved = False
        # Adaptive step size: decrease as we converge
        eps = base_eps * (0.85 ** it)
        eps = max(eps, 0.01)
        
        for i in range(n_exercise):
            val_curr = compute_policy_value(thresholds)
            
            # Try +eps
            thresh_plus = thresholds.copy()
            thresh_plus[i] = min(thresh_plus[i] + eps, 2.0 * K)  # Upper bound
            val_plus = compute_policy_value(thresh_plus)
            
            # Try -eps
            thresh_minus = thresholds.copy()
            thresh_minus[i] = max(thresh_minus[i] - eps, 0.0)  # Lower bound
            val_minus = compute_policy_value(thresh_minus)
            
            # Choose the best
            best_val = val_curr
            best_thresh = thresholds.copy()
            
            if val_plus > best_val:
                best_val = val_plus
                best_thresh = thresh_plus
                improved = True
            if val_minus > best_val:
                best_val = val_minus
                best_thresh = thresh_minus
                improved = True
            
            thresholds = best_thresh
        
        if not improved:
            break
    
    # Final polish with smaller steps
    for it in range(20):
        improved = False
        eps = 0.1 * (0.9 ** it)
        eps = max(eps, 0.005)
        
        for i in range(n_exercise):
            val_curr = compute_policy_value(thresholds)
            
            thresh_plus = thresholds.copy()
            thresh_plus[i] = min(thresh_plus[i] + eps, 2.0 * K)
            val_plus = compute_policy_value(thresh_plus)
            
            thresh_minus = thresholds.copy()
            thresh_minus[i] = max(thresh_minus[i] - eps, 0.0)
            val_minus = compute_policy_value(thresh_minus)
            
            best_val = val_curr
            best_thresh = thresholds.copy()
            
            if val_plus > best_val:
                best_val = val_plus
                best_thresh = thresh_plus
                improved = True
            if val_minus > best_val:
                best_val = val_minus
                best_thresh = thresh_minus
                improved = True
            
            thresholds = best_thresh
        
        if not improved:
            break
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, exercise_times, K):
            self.thresholds = thresholds
            self.exercise_times = exercise_times
            self.K = K
            self.n_exercise = len(thresholds)
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            
            Parameters:
            - current_state: array of current spot prices (one per path)
            - time: current time (scalar)
            
            Returns:
            - boolean vector indicating whether to exercise for each path
            """
            if isinstance(time, (int, float, np.floating)):
                t_val = float(time)
            elif hasattr(time, '__len__') and len(time) > 0:
                t_val = float(time[0])
            else:
                t_val = float(time)
            
            # Find the exercise date index closest to time
            idx = int(np.argmin(np.abs(self.exercise_times - t_val)))
            
            # Check if it's close enough to an exercise date
            if abs(self.exercise_times[idx] - t_val) > 0.02:
                return np.zeros(len(current_state), dtype=bool)
            
            # Exercise if S < threshold
            return current_state < self.thresholds[idx]
    
    return ExercisePolicy(thresholds, exercise_times, K)
# EVOLVE-BLOCK-END
