"""Rewritten program using direct Monte Carlo optimization of exercise thresholds."""
# EVOLVE-BLOCK-START
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Monte Carlo estimation of
    continuation values and threshold optimization.
    
    The policy at each exercise date t_i is: exercise if S_t < threshold_i.
    We estimate optimal thresholds by comparing immediate exercise payoff
    to the estimated continuation value at each exercise date.
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
    # S[i] is the spot price at exercise date i for all paths
    S = np.zeros((n_exercise, n_paths))
    for i in range(n_exercise):
        S[i] = training_paths[:, exercise_cols[i]]
    
    # Initialize thresholds
    thresholds = np.zeros(n_exercise)
    
    # For the last exercise date, the optimal threshold is K (exercise if in-the-money)
    # because there's no continuation value
    thresholds[-1] = K
    
    # For earlier dates, we estimate the continuation value and find the threshold
    # where immediate payoff = continuation value
    # 
    # Key insight: For a given threshold at date i, paths with S_i < threshold[i]
    # exercise and get payoff (K - S_i). Paths with S_i >= threshold[i] continue.
    # The continuation value for a path that continues at date i is the discounted
    # value of what happens at date i+1 (and beyond).
    #
    # We can estimate this by: for paths that continue at date i (S_i >= threshold[i]),
    # look at their value at date i+1 under the policy, and discount back.
    #
    # This is a fixed-point iteration: start with an initial guess, compute continuation
    # values, update thresholds, repeat.
    
    n_iter = 50
    convergence_tol = 1e-4
    
    for iteration in range(n_iter):
        old_thresholds = thresholds.copy()
        
        # Process from last exercise date to first
        # cont_value[i] will be an array of continuation values for each path at date i
        # For the last date, continuation value is 0 (option expires if not exercised)
        
        # We'll compute the "value at date i" for each path under the current policy
        # value_at_date[i, j] = the value of the option at date i for path j
        # If path j exercises at date i: value = K - S_i
        # If path j continues at date i: value = discounted expected future value
        
        # Start from the last exercise date
        # At the last date, if you don't exercise, the option expires worthless
        # So the value at the last date is: max(K - S_{n-1}, 0) if we exercise, 0 if not
        # But with threshold policy: exercise if S < threshold
        # value_at_date[n-1, j] = (K - S[n-1, j]) if S[n-1, j] < threshold[n-1] else 0
        
        value_at_date = np.zeros((n_exercise, n_paths))
        
        # Last exercise date
        for j in range(n_paths):
            if S[-1, j] < thresholds[-1]:
                value_at_date[-1, j] = max(K - S[-1, j], 0.0)
            else:
                value_at_date[-1, j] = 0.0
        
        # Now work backward
        for i in range(n_exercise - 2, -1, -1):
            # Discount factor from date i+1 to date i
            dt_next = exercise_times[i+1] - exercise_times[i]
            disc_i = np.exp(-r * dt_next)
            
            # For each path, if it continues at date i, its value is disc_i * value_at_date[i+1, j]
            # But we need to estimate this properly.
            # 
            # The issue: value_at_date[i+1, j] was computed assuming path j's state at i+1
            # is what it actually is. This is a pathwise continuation value, which is an
            # unbiased estimate of the conditional expectation if the paths are i.i.d.
            #
            # However, for a threshold policy, the continuation value at date i for a path
            # with state S_i should be the EXPECTED value of future payoffs given S_i.
            # Using the actual future state on the same path is a valid Monte Carlo estimate
            # of this conditional expectation (since paths are i.i.d.).
            #
            # So: cont_value[i, j] = disc_i * value_at_date[i+1, j]
            # This is the estimated continuation value for path j at date i.
            
            cont_value_i = disc_i * value_at_date[i+1, :]
            
            # Now update value_at_date[i]:
            # If S_i < threshold[i]: exercise, value = K - S_i
            # If S_i >= threshold[i]: continue, value = cont_value_i[j]
            
            for j in range(n_paths):
                if S[i, j] < thresholds[i]:
                    value_at_date[i, j] = max(K - S[i, j], 0.0)
                else:
                    value_at_date[i, j] = cont_value_i[j]
            
            # Now estimate the optimal threshold for date i:
            # The optimal threshold is where (K - S_i) = cont_value_i
            # i.e., S_i = K - cont_value_i
            # 
            # We want to find the threshold such that paths with S_i < threshold exercise
            # (because K - S_i > cont_value) and paths with S_i >= threshold continue
            # (because K - S_i <= cont_value).
            #
            # The optimal threshold is the value of S_i where K - S_i = cont_value_i.
            # Since cont_value_i varies by path, we need to find a single threshold.
            # 
            # A good estimate: the median of (K - cont_value_i) for paths that are
            # in a "boundary" region, or simply the average.
            # 
            # Actually, the optimal threshold theta* satisfies:
            # P(S_i < theta* | S_i < theta*) * E[K - S_i | S_i < theta*] + 
            # P(S_i >= theta* | S_i < theta*) * E[cont_value_i | S_i >= theta*]
            # vs other thresholds.
            #
            # Simpler: the optimal threshold is approximately the value where
            # K - S_i = cont_value_i on average.
            # 
            # Let's compute the implied threshold for each path: theta_j = K - cont_value_i[j]
            # Then the optimal threshold is a robust estimate of these values.
            
            implied_thresholds = K - cont_value_i
            # Filter to reasonable range
            valid = (implied_thresholds > 0) & (implied_thresholds < K * 2)
            if np.any(valid):
                # Use the median for robustness
                thresholds[i] = np.median(implied_thresholds[valid])
            else:
                thresholds[i] = K
        
        # Check convergence
        if np.max(np.abs(thresholds - old_thresholds)) < convergence_tol:
            break
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, exercise_cols, K, exercise_times):
            self.thresholds = thresholds
            self.exercise_cols = exercise_cols
            self.K = K
            self.n_exercise = len(thresholds)
            self.exercise_times = exercise_times
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            
            Parameters:
            - current_state: array of current spot prices (one per path)
            - time: current time (scalar)
            
            Returns:
            - boolean vector indicating whether to exercise for each path
            """
            t_val = float(time)
            # Find the exercise date index closest to time
            idx = np.argmin(np.abs(self.exercise_times - t_val))
            # Check if it's close enough
            if abs(self.exercise_times[idx] - t_val) > 0.01:
                return np.zeros(len(current_state), dtype=bool)
            
            # Exercise if S < threshold
            return current_state < self.thresholds[idx]
    
    return ExercisePolicy(thresholds, exercise_cols, K, exercise_times)
# EVOLVE-BLOCK-END
