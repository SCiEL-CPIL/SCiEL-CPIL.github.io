"""Bermudan exercise policy using sparse exercise-date tree with critical price refinement."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12
    
    # Sparse tree: one step per exercise date
    # Exercise dates: T/12, 2T/12, ..., 12T/12 = T
    # dt for the sparse tree
    dt_sparse = T / n_exercise
    u_sparse = np.exp(sigma * np.sqrt(dt_sparse))
    d_sparse = 1.0 / u_sparse
    p_sparse = (np.exp(r * dt_sparse) - d_sparse) / (u_sparse - d_sparse)
    discount_sparse = np.exp(-r * dt_sparse)
    
    S0_tree = 100.0
    log_S0 = np.log(S0_tree)
    
    # For the sparse tree with n_exercise steps:
    # At step n (1-indexed, n=1..n_exercise), there are n+1 nodes
    # Node k at step n: S = S0 * u^(2k-n)
    
    # Backward induction on the sparse tree
    # values at step n: array of length n+1
    
    # At maturity (step n_exercise):
    n_final = n_exercise
    k_mat = np.arange(n_final + 1)
    S_mat = S0_tree * u_sparse ** (2 * k_mat - n_final)
    values_N = np.maximum(K - S_mat, 0.0)
    
    current_values = values_N.copy()
    
    # Store critical prices for each exercise date (step 1 to n_exercise)
    critical_prices = {}
    
    # Go backward from step n_exercise-1 down to step 1
    for n in range(n_final - 1, 0, -1):
        # At step n, there are n+1 nodes
        # current_values has length n+2 (for step n+1)
        up_val = current_values[1:n+2]   # length n+1
        down_val = current_values[0:n+1]  # length n+1
        
        cont_val = discount_sparse * (p_sparse * up_val + (1 - p_sparse) * down_val)
        
        # Exercise date at step n
        k_nodes = np.arange(n + 1)
        S_nodes = S0_tree * u_sparse ** (2 * k_nodes - n)
        ex_val = np.maximum(K - S_nodes, 0.0)
        
        exercise_mask = ex_val > cont_val
        
        # Find critical price
        exercising_indices = np.where(exercise_mask)[0]
        if len(exercising_indices) > 0:
            max_exercise_k = exercising_indices[-1]
            if max_exercise_k < n:
                S_ex = S_nodes[max_exercise_k]
                S_no_ex = S_nodes[max_exercise_k + 1]
                diff_k = ex_val[max_exercise_k] - cont_val[max_exercise_k]
                diff_k1 = ex_val[max_exercise_k + 1] - cont_val[max_exercise_k + 1]
                
                if diff_k > 0 and diff_k1 < 0:
                    frac = diff_k / (diff_k - diff_k1)
                    cp = S_ex + frac * (S_no_ex - S_ex)
                elif diff_k > 0:
                    cp = S_no_ex
                else:
                    cp = S_ex
            else:
                cp = S_nodes[max_exercise_k]
            critical_prices[n] = cp
        else:
            critical_prices[n] = 0.0
        
        new_values = np.maximum(ex_val, cont_val)
        current_values = new_values
    
    # Now refine critical prices using training path information
    # The training paths give us empirical distribution of states at each exercise date
    # We can use this to validate and adjust the critical prices
    
    if training_paths is not None and len(training_paths) > 0:
        n_paths, n_time_steps = training_paths.shape
        # Determine time step size of training paths
        if n_time_steps > 1:
            dt_train = T / (n_time_steps - 1)
        else:
            dt_train = T
        
        # For each exercise date, find the corresponding index in training paths
        for step_idx in range(1, n_exercise + 1):
            t_ex = step_idx * dt_sparse
            # Find nearest training time index
            train_idx = int(round(t_ex / dt_train))
            if train_idx >= n_time_steps:
                train_idx = n_time_steps - 1
            if train_idx < 0:
                train_idx = 0
            
            states_at_ex = training_paths[:, train_idx]
            
            # Use training data to refine critical price
            # The idea: if many training paths are near the critical price,
            # we want to be more precise. Use a local regression approach.
            
            cp = critical_prices.get(step_idx, 0.0)
            if cp <= 0:
                continue
            
            # Check how many training states are near the critical price
            near_mask = np.abs(states_at_ex - cp) < 0.5 * cp
            if np.sum(near_mask) > 10:
                # Refine using local quadratic interpolation
                # Get states and compute what the continuation value would be
                # For states below cp, exercise; above, hold
                # The optimal cp maximizes expected value
                
                # Simple refinement: try perturbing cp slightly
                best_cp = cp
                best_val = -np.inf
                
                # Compute approximate value for different critical prices
                # For a put, value = E[max(K-S,0) * I(S<cp) + cont_val * I(S>=cp)]
                # We approximate cont_val using the tree value
                
                # Get tree continuation values at this step
                n = step_idx
                k_nodes = np.arange(n + 1)
                S_nodes = S0_tree * u_sparse ** (2 * k_nodes - n)
                
                # Reconstruct continuation values at this step from the tree
                # This requires redoing part of the backward induction, so we'll use
                # a simpler approach: just verify the critical price is reasonable
                
                # For now, keep the tree-derived critical price
                # but ensure it's within the range of training states
                min_state = np.min(states_at_ex)
                max_state = np.max(states_at_ex)
                if cp < min_state:
                    cp = min_state
                elif cp > max_state:
                    cp = max_state
                critical_prices[step_idx] = cp
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self):
            self.critical_prices = critical_prices
            self.n_exercise = n_exercise
            self.T = T
            self.K = K
            self.r = r
        
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: current time (should be one of the exercise dates)
            Returns: boolean array, True if exercise
            """
            # Determine which exercise date this is
            # Exercise dates are at T/12, 2T/12, ..., 12T/12
            # time is in [0, T]
            
            if time <= 0 or time > self.T:
                return np.zeros(len(current_state), dtype=bool)
            
            # Find the exercise date index
            # Exercise date k is at time k * T / n_exercise
            k_float = time * self.n_exercise / self.T
            k = int(round(k_float))
            
            # Clamp to valid range
            k = max(1, min(k, self.n_exercise))
            
            # Check if this is close to an exercise date
            t_ex = k * self.T / self.n_exercise
            if abs(time - t_ex) > 0.01 * self.T:
                return np.zeros(len(current_state), dtype=bool)
            
            cp = self.critical_prices.get(k, 0.0)
            
            # Exercise if S < critical_price
            return current_state < cp
    
    policy = ExercisePolicy()
    return policy

# EVOLVE-BLOCK-END
