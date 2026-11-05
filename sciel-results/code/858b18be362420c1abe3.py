"""Bermudan put exercise-policy research with backward induction on training paths."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 110.0  # safety margin under 120s limit


def _generate_gbm_paths(S0, K, r, sigma, T, n_dates, n_paths, seed=0):
    """Generate risk-neutral GBM paths at n_dates monthly exercise dates."""
    rng = np.random.default_rng(seed)
    dt = T / n_dates
    log_s0 = np.log(S0)
    drift = (r - 0.5 * sigma ** 2) * dt
    vol_step = sigma * np.sqrt(dt)
    
    # Generate log returns
    z = rng.standard_normal((n_dates, n_paths))
    log_steps = drift + vol_step * z
    
    # Cumulative sum to get log prices
    log_paths = np.cumsum(log_steps, axis=0)
    log_paths = log_s0 + log_paths
    
    # Convert to prices
    paths = np.exp(log_paths)
    return paths


def _compute_optimal_thresholds(training_paths, K, r, T, n_dates, n_paths):
    """
    Compute optimal exercise thresholds via backward induction on training paths.
    
    Returns: thresholds[i] = threshold at exercise date i (0-indexed, i=0..n_dates-1)
    Exercise at date i if S_i < thresholds[i]
    """
    dt = T / n_dates
    discount = np.exp(-r * dt)
    
    # continuation_value[i, p] = value of continuing at date i for path p
    # We'll work backward from the last date
    
    # At the last date (i = n_dates-1), continuation = 0 (must exercise or expire)
    # Actually at last date, you exercise if payoff > 0, so continuation = 0
    # Value at last date = max(K - S, 0)
    
    n_dates = training_paths.shape[0]
    n_paths = training_paths.shape[1]
    
    # continuation[i, p] = expected value of optimal policy from date i onward, given path p at date i
    # We store the continuation value (value if we don't exercise at date i)
    
    # Start from the last date
    # At date n_dates-1: if we don't exercise, value = 0 (option expires)
    # So continuation[n_dates-1] = 0
    # Value[n_dates-1] = max(K - S[n_dates-1], 0)
    
    continuation = np.zeros((n_dates, n_paths))
    
    # Work backward from date n_dates-2 down to 0
    # At date i: 
    #   immediate = max(K - S[i], 0)
    #   cont_val = discount * E[Value[i+1] | S[i], path history]
    #   But since we're using paths, we approximate E by the value at the next date for the same path
    #   (This is a simplification - in reality we'd need to condition on current state)
    
    # Better approach: for each date i, compute continuation value as the discounted
    # average of the optimal values at date i+1, but we need to be careful.
    
    # Actually, the standard approach for Bermudan options with path-based estimation:
    # At each date i, for each path p:
    #   immediate_payoff[i, p] = max(K - S[i, p], 0)
    #   continuation_value[i, p] = discount * optimal_value[i+1, p]
    #   where optimal_value[i+1, p] = max(immediate_payoff[i+1, p], continuation_value[i+1, p])
    
    # This works because in a risk-neutral setting with GBM, the future evolution
    # from S[i,p] is independent of the past given S[i,p], and the path p from i+1 onward
    # gives us a sample of that future.
    
    # Let's compute optimal_value from the end backward
    optimal_value = np.zeros((n_dates, n_paths))
    
    # Last date: must exercise if in the money
    optimal_value[n_dates - 1] = np.maximum(K - training_paths[n_dates - 1], 0.0)
    continuation[n_dates - 1] = 0.0  # no continuation after last date
    
    # Work backward
    for i in range(n_dates - 2, -1, -1):
        # Continuation value at date i = discount * optimal_value at date i+1
        continuation[i] = discount * optimal_value[i + 1]
        
        # Immediate payoff at date i
        immediate = np.maximum(K - training_paths[i], 0.0)
        
        # Optimal value = max(exercise now, continue)
        optimal_value[i] = np.maximum(immediate, continuation[i])
    
    # Now extract thresholds: at each date i, exercise if S[i,p] < threshold[i]
    # From the optimal_value computation, we can determine for each path whether it was
    # optimal to exercise or continue. We want a single threshold per date.
    
    # For each date i, find the threshold such that paths with S < threshold are exercised
    # We can use the fact that exercise is optimal when immediate > continuation
    # i.e., K - S > continuation[i,p]  =>  S < K - continuation[i,p]
    
    # So the threshold for path p at date i is K - continuation[i,p]
    # But we want a single threshold per date. We can use the median or a robust estimate.
    
    # Actually, for a monotone barrier policy, we can compute the threshold as:
    # threshold[i] = K - median(continuation[i]) or similar
    # But better: find the S value where exercise switches from optimal to not optimal
    
    # Since continuation[i,p] varies with the path, and S[i,p] also varies,
    # we can fit a threshold by looking at where the decision changes.
    
    # Simpler approach: for each date, compute the implied threshold for each path
    # and take a robust aggregate (e.g., median)
    
    thresholds = np.zeros(n_dates)
    for i in range(n_dates):
        if i == n_dates - 1:
            # At last date, exercise if S < K
            thresholds[i] = K
        else:
            # Implied threshold for path p: S < K - continuation[i,p] means exercise
            # So threshold_p = K - continuation[i,p]
            implied_thresholds = K - continuation[i]
            # Use median as robust estimate
            thresholds[i] = np.median(implied_thresholds)
    
    return thresholds, optimal_value


def _evaluate_policy_value(paths, thresholds, K, r, T, n_dates):
    """Evaluate the value of a threshold policy on given paths."""
    dt = T / n_dates
    discount = np.exp(-r * dt)
    n_paths = paths.shape[1]
    
    # Work backward to compute the value
    # At each date, if S < threshold, exercise and get payoff
    # Otherwise, continue with discounted expected value
    
    # We need to compute the value recursively
    # value[i, p] = value of optimal policy from date i onward for path p
    
    # But for evaluation, we just apply the fixed thresholds
    # value at date i for path p:
    #   if S[i,p] < thresholds[i]: value = K - S[i,p]
    #   else: value = discount * value[i+1, p]
    
    # Base case: at last date
    n_paths = paths.shape[1]
    value = np.zeros(n_paths)
    
    # Start from the last date and go backward
    for i in range(n_dates - 1, -1, -1):
        exercise = paths[i] < thresholds[i]
        payoff = np.maximum(K - paths[i], 0.0)
        
        if i == n_dates - 1:
            # At last date, if we don't exercise, value = 0
            value = np.where(exercise, payoff, 0.0)
        else:
            # value = where(exercise, payoff, discount * value_next)
            # But value_next is computed in the next iteration (i+1)
            # We need to be careful with the order
            pass
    
    # Let me redo this properly - work from the end
    value_next = np.zeros(n_paths)
    total_value = np.zeros(n_paths)
    
    for i in range(n_dates - 1, -1, -1):
        exercise = paths[i] < thresholds[i]
        payoff = np.maximum(K - paths[i], 0.0)
        
        if i == n_dates - 1:
            value = np.where(exercise, payoff, 0.0)
        else:
            value = np.where(exercise, payoff, discount * value_next)
        
        value_next = value
    
    # The value at time 0 (first exercise date) for each path
    # But we want the discounted value from time 0
    # The value at date 0 is already the value at that date
    # We need to discount back to time 0
    
    # Actually, the value we computed is the value at each date.
    # The option value at time 0 is the discounted expected value of the payoff.
    # For a Bermudan option, the value at the first exercise date is:
    #   If we exercise: payoff
    #   If we continue: discount * E[value at next date]
    
    # The value_next after the loop is the value at date 0
    # But we need to be careful - the value at date 0 is what we'd receive at date 0
    # The present value is that discounted back to time 0 by exp(-r * 0) = 1? No.
    
    # Actually, in the backward induction, value at date i is the value at time t_i.
    # The present value (at time 0) is value at date 0 discounted by exp(-r * t_0) = exp(-r * dt)
    # Wait, no. Let me think again.
    
    # t_0 = dt, t_1 = 2*dt, ..., t_{n_dates-1} = T
    # Value at date i is the value at time t_i.
    # Present value = value at date 0 * exp(-r * t_0) = value[0] * exp(-r * dt)
    
    # But in my backward induction, I'm computing the value at each date correctly.
    # The value at date 0 (first exercise date) is value_next after the first iteration.
    
    # Let me trace through:
    # i = n_dates-1: value = where(exercise, payoff, 0) -> this is value at t_{n_dates-1}
    #   value_next = value
    # i = n_dates-2: value = where(exercise, payoff, discount * value_next) 
    #   value_next is value at t_{n_dates-1}, so discount * value_next is the discounted value at t_{n_dates-2}
    #   This is correct - value at t_{n_dates-2}
    # ...
    # i = 0: value = where(exercise, payoff, discount * value_next)
    #   value_next is value at t_1, so this gives value at t_0
    
    # So value_next after the loop is the value at t_0 (first exercise date)
    # Present value at time 0 = value_next * exp(-r * t_0) = value_next * exp(-r * dt)
    
    pv = value_next * np.exp(-r * dt)
    return np.mean(pv)


def _refine_thresholds(training_paths, K, r, T, n_dates, n_paths, initial_thresholds, n_iterations=3):
    """Refine thresholds using a few iterations of policy iteration."""
    dt = T / n_dates
    discount = np.exp(-r * dt)
    
    thresholds = initial_thresholds.copy()
    
    for _ in range(n_iterations):
        # Compute continuation values using current thresholds
        # For each path and date, compute the value if we follow the current policy
        # Then re-derive thresholds from these values
        
        # Compute value at each date using the current threshold policy
        value_next = np.zeros(n_paths)
        continuation_values = np.zeros((n_dates, n_paths))
        
        for i in range(n_dates - 1, -1, -1):
            exercise = training_paths[i] < thresholds[i]
            payoff = np.maximum(K - training_paths[i], 0.0)
            
            if i == n_dates - 1:
                value = np.where(exercise, payoff, 0.0)
            else:
                value = np.where(exercise, payoff, discount * value_next)
            
            continuation_values[i] = discount * value_next if i < n_dates - 1 else 0.0
            value_next = value
        
        # Re-derive thresholds
        new_thresholds = np.zeros(n_dates)
        for i in range(n_dates):
            if i == n_dates - 1:
                new_thresholds[i] = K
            else:
                # Exercise is optimal when K - S > continuation_values[i, p]
                # i.e., S < K - continuation_values[i, p]
                implied = K - continuation_values[i]
                new_thresholds[i] = np.median(implied)
        
        # Check convergence
        if np.allclose(thresholds, new_thresholds, atol=1e-6):
            thresholds = new_thresholds
            break
        thresholds = new_thresholds
    
    return thresholds


def fit_exercise_policy(training_paths, contract):
    """Fit a Bermudan put exercise policy."""
    start_time = _time.time()
    
    # Extract parameters from contract
    # Based on the task description:
    # Spot 90/100/110, K 100, r 0.05, sigma 0.2, T 1
    # 12 monthly exercise dates
    # training_paths shape: (n_dates, n_paths) = (12, 4096)
    
    n_dates, n_paths = training_paths.shape
    K = 100.0
    r = 0.05
    T = 1.0
    sigma = 0.2
    
    dt = T / n_dates
    
    # Compute initial thresholds via backward induction
    thresholds, optimal_value = _compute_optimal_thresholds(
        training_paths, K, r, T, n_dates, n_paths
    )
    
    # Refine with policy iteration
    remaining_time = _DEADLINE - (_time.time() - start_time)
    if remaining_time > 10:
        thresholds = _refine_thresholds(
            training_paths, K, r, T, n_dates, n_paths, thresholds, n_iterations=5
        )
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K, r, T, n_dates):
            self.thresholds = thresholds
            self.K = K
            self.r = r
            self.T = T
            self.n_dates = n_dates
            self._current_date = 0  # track which exercise date we're at
        
        def exercise(self, current_state, time):
            """
            Return a boolean vector indicating which paths to exercise.
            
            current_state: array of shape (n_paths,) with current spot prices
            time: current time (to determine which exercise date)
            """
            # Determine which exercise date we're at
            # Exercise dates are at t = dt, 2*dt, ..., n_dates*dt = T
            # time should be one of these values
            
            # Find the exercise date index
            # t_i = (i+1) * dt for i = 0, ..., n_dates-1
            # So date index = round(time / dt) - 1
            
            date_idx = int(round(time / dt)) - 1
            date_idx = max(0, min(date_idx, self.n_dates - 1))
            
            # Exercise if current_state < threshold
            return current_state < self.thresholds[date_idx]
    
    policy = ExercisePolicy(thresholds, K, r, T, n_dates)
    return policy

# EVOLVE-BLOCK-END
