"""Bermudan exercise-policy research: direct Monte Carlo optimization of exercise thresholds."""
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    num_steps = 12
    
    # Ensure training_paths is 2D
    training_paths = np.asarray(training_paths, dtype=np.float64)
    if training_paths.ndim == 1:
        training_paths = training_paths.reshape(1, -1)
    
    N, total_cols = training_paths.shape
    
    # Determine which columns are exercise dates
    # If total_cols == 13, columns 1..12 are exercise dates (t=0 is column 0)
    # If total_cols == 12, columns 0..11 are exercise dates
    if total_cols == num_steps + 1:
        # Columns 1..12 are the 12 exercise dates
        S = training_paths[:, 1:num_steps+1]  # shape (N, 12)
    elif total_cols == num_steps:
        # Columns 0..11 are the 12 exercise dates
        S = training_paths[:, 0:num_steps]
    else:
        # Fallback: use last num_steps columns
        S = training_paths[:, -num_steps:]
    
    N, M = S.shape  # M should be 12
    
    # Discount factors for each exercise date
    dt = T / num_steps
    # Time of exercise date i is (i+1)*dt for i=0..11
    times = np.arange(1, num_steps+1) * dt
    discount = np.exp(-r * times)  # shape (12,)
    
    # Payoff if exercised at date i: max(K - S_i, 0) * discount[i]
    # We want to maximize E[sum_i 1{exercise at i} * payoff_i]
    # Under optimal policy, at each date we exercise if continuation value < immediate payoff
    
    # Initialize thresholds: start with simple heuristic
    # For a put, we exercise when S is low. Initial threshold for date i
    # Use a simple approximation: threshold_i = K * exp(-something)
    # Let's initialize thresholds at K for all dates, then optimize
    thresholds = np.full(num_steps, K, dtype=np.float64)
    
    # Stochastic gradient ascent on thresholds
    # Objective: mean over paths of discounted payoff under threshold policy
    # Policy: at date i, exercise if S_i < threshold_i (for put)
    # But we need to account for the fact that once you exercise, you stop
    # So the policy is: at date i, exercise if S_i < threshold_i AND you haven't exercised before
    
    def compute_objective(thr):
        """Compute mean discounted payoff for given thresholds."""
        N_local = S.shape[0]
        total_payoff = np.zeros(N_local, dtype=np.float64)
        exercised = np.zeros(N_local, dtype=bool)
        
        for i in range(num_steps):
            # At date i, exercise if not yet exercised and S_i < threshold_i
            should_exercise = (~exercised) & (S[:, i] < thr[i])
            if np.any(should_exercise):
                total_payoff[should_exercise] += (K - S[should_exercise, i]) * discount[i]
                exercised[should_exercise] = True
        
        return np.mean(total_payoff)
    
    # Compute initial objective
    best_obj = compute_objective(thresholds)
    best_thr = thresholds.copy()
    
    # Stochastic gradient ascent
    # Use a smooth approximation: replace hard threshold with sigmoid
    # exercise_prob_i = sigmoid((thr_i - S_i) / temperature)
    # This makes the objective differentiable
    
    temperature = 0.5
    learning_rate = 0.1
    batch_size = min(512, N)
    
    max_iterations = 2000
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > DEADLINE - 5.0:
            break
        
        # Random batch
        if batch_size < N:
            batch_idx = np.random.choice(N, batch_size, replace=False)
        else:
            batch_idx = np.arange(N)
        
        S_batch = S[batch_idx]
        B = S_batch.shape[0]
        
        # Compute objective and gradient using sigmoid smoothing
        total_payoff = np.zeros(B, dtype=np.float64)
        grad = np.zeros(num_steps, dtype=np.float64)
        
        # Track probability of being active (not yet exercised) at each step
        active_prob = np.ones(B, dtype=np.float64)
        
        for i in range(num_steps):
            # Probability of exercising at step i given active
            z = (thresholds[i] - S_batch[:, i]) / temperature
            # Sigmoid
            p_exercise = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            
            # Expected payoff contribution
            expected_exercise = active_prob * p_exercise
            payoff_i = (K - S_batch[:, i]) * discount[i]
            total_payoff += expected_exercise * payoff_i
            
            # Update active probability
            active_prob *= (1.0 - p_exercise)
            
            # Gradient with respect to thresholds[i]
            # d/dthr_i of sigmoid((thr_i - S_i)/temp) = sigmoid * (1-sigmoid) / temp
            sp = p_exercise * (1.0 - p_exercise) / temperature
            # d objective / d thr_i = sum over paths of active_prob * sp * payoff_i
            # But we also need to account for the effect on future steps through active_prob
            # For simplicity, use only the direct effect (ignoring future interactions)
            # This is a common approximation in stochastic gradient methods
            grad[i] += np.sum(active_prob * sp * payoff_i)
        
        obj = np.mean(total_payoff)
        
        if obj > best_obj:
            best_obj = obj
            best_thr = thresholds.copy()
        
        # Update thresholds
        thresholds += learning_rate * grad / B
        # Keep thresholds in reasonable range
        thresholds = np.clip(thresholds, 1.0, 200.0)
        
        # Decay learning rate
        learning_rate *= 0.999
        temperature *= 0.9995
        temperature = max(temperature, 0.01)
        
        iteration += 1
    
    # Use the best thresholds found
    final_thresholds = best_thr
    
    # Create policy object
    class BermudanPolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K
        
        def exercise(self, current_state, time_index):
            """
            current_state: array of current spot prices, shape (N,)
            time_index: integer index of the current exercise date (0-based, 0..11)
            Returns: boolean vector indicating whether to exercise
            """
            if time_index is None:
                time_index = 0
            time_index = int(time_index)
            if time_index < 0 or time_index >= len(self.thresholds):
                return np.zeros(len(current_state), dtype=bool)
            
            current_state = np.asarray(current_state, dtype=np.float64)
            return current_state < self.thresholds[time_index]
    
    return BermudanPolicy(final_thresholds, K)
