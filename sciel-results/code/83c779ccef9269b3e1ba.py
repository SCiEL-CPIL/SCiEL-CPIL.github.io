"""Bermudan put exercise policy via Monte Carlo Backward Induction with Quantile Binning."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Ensure n_exercise is integer
    n_exercise = int(n_exercise)
    dt = T / n_exercise
    disc = np.exp(-r * dt)

    # training_paths shape: (n_paths, n_exercise)
    # We assume the columns correspond to exercise dates 1..n_exercise
    if training_paths is None or len(training_paths) == 0:
        # Fallback: Generate paths if not provided (though contract says it is)
        n_paths = 4096
        z = np.random.standard_normal((n_paths, n_exercise))
        training_paths = S0 * np.exp((r - 0.5 * sigma**2) * np.arange(1, n_exercise+1) * dt + sigma * np.sqrt(np.arange(1, n_exercise+1) * dt) * z)
        
    n_paths = training_paths.shape[0]
    
    # Number of bins for value function approximation
    n_bins = 200
    
    # Storage for policy
    # For each step, we store bin edges and the exercise decision for each bin
    policy_data = [None] * n_exercise
    
    # V is the value at the current step for each path
    # Start at the last exercise date (index n_exercise - 1)
    S_last = training_paths[:, -1]
    payoff_last = np.maximum(K - S_last, 0.0)
    V = payoff_last.copy()
    
    # At the last step, we always exercise if in the money (European terminal condition)
    # But the policy asks for exercise at step n_exercise.
    # The decision at step n is simply: Exercise if Payoff > 0 (since continuation is 0)
    # Actually, at T, it's European. So V_T = Payoff.
    # The exercise decision at T is Payoff > 0.
    
    # We need to store the decision for step n_exercise (index -1)
    # But the loop goes backwards from n_exercise-1 down to 0.
    # So we handle step n_exercise separately or as the base case.
    
    # Let's define step indices 0 to n_exercise-1 corresponding to t=dt, ..., t=T.
    # training_paths[:, 0] is S at t=dt.
    # training_paths[:, n_exercise-1] is S at t=T.
    
    # Base case: Step n_exercise-1 (last step)
    # Value is Payoff.
    # Decision: Exercise if Payoff >= Continuation (which is 0). So Payoff > 0.
    S_curr = training_paths[:, n_exercise-1]
    edges = np.quantile(S_curr, np.linspace(0, 1, n_bins + 1))
    # Ensure unique edges for searchsorted
    # Add small epsilon to handle duplicates if any
    edges[1:] += 1e-9
    decisions = (np.maximum(K - S_curr, 0.0) > 0)
    
    # We store the decision for each bin.
    # Since we use quantiles, each bin has roughly n_paths/n_bins samples.
    # We calculate the mean decision? No, the decision is deterministic given S.
    # But S varies within a bin.
    # The policy for a *new* spot S is: Exercise if S is in the region where the optimal policy says so.
    # In backward induction, the optimal policy is: Exercise if Payoff(S) >= E[disc * V_next | S].
    # Within a bin, we approximate E[disc * V_next | S] by the mean of disc * V_next for paths in that bin.
    # And Payoff(S) is known analytically.
    # So for a bin, we don't have a single "Payoff".
    # However, we can store the *Threshold* or simply the *Value Function* approximation.
    
    # Better approach:
    # Store the approximation of the Continuation Value C(S) for each bin.
    # Then in `exercise`, compute Payoff(S) and compare with C(S) interpolated from bins.
    # This is more robust than storing a boolean decision which depends on the specific S in the bin.
    
    policy_data = [None] * n_exercise
    
    # We need to store:
    # 1. Bin edges (same for all steps if we use fixed grid, but quantiles change per step)
    # 2. Continuation value approximation per bin (mean of disc * V_next)
    
    # Let's restart the backward induction properly.
    
    V_next = np.maximum(K - training_paths[:, n_exercise-1], 0.0) # V at T
    
    # Step n_exercise-1 (last, index n_exercise-1)
    # At this step, Continuation is 0.
    # So V = Payoff.
    # We store the bin edges and the "Continuation Value" (which is 0) ?
    # No, at the last step, the decision is Payoff > 0.
    # For consistency, let's store the *Value Function* V(S) approximation.
    # V(S) = max(Payoff(S), C(S)).
    # If we store V(S), we can't easily decide "Exercise" vs "Continue" without knowing C(S).
    # So we MUST store C(S) or the decision boundary.
    
    # Let's store C(S) approximation per bin.
    # And in `exercise`, we compute Payoff(S) and compare.
    
    # Step n_exercise-1:
    S_curr = training_paths[:, n_exercise-1]
    edges = np.quantile(S_curr, np.linspace(0, 1, n_bins + 1))
    edges[1:] += 1e-9
    # C(S) = 0 for all S at the last step (European)
    C_vals = np.zeros(n_bins)
    policy_data[n_exercise-1] = (edges, C_vals)
    
    # Backward induction from step n_exercise-2 down to 0
    for step in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break
            
        S_curr = training_paths[:, step]
        S_next = training_paths[:, step + 1]
        
        # Calculate bins for S_curr
        edges = np.quantile(S_curr, np.linspace(0, 1, n_bins + 1))
        edges[1:] += 1e-9
        
        # Determine bin index for each path
        bin_indices = np.searchsorted(edges, S_curr, side='left')
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # We need E[disc * V_next | S_curr]
        # V_next is the value at step+1, which we have calculated in the previous iteration.
        # Wait, V_next is an array of values for each path.
        # We need to aggregate V_next by the bin of S_curr.
        
        # Initialize sum and count for each bin
        sum_V = np.zeros(n_bins)
        count_V = np.zeros(n_bins)
        
        np.add.at(sum_V, bin_indices, V_next)
        np.add.at(count_V, bin_indices, 1)
        
        # Avoid division by zero
        count_V_safe = np.where(count_V == 0, 1, count_V)
        mean_V = sum_V / count_V_safe
        
        # Continuation value C(S) = disc * E[V_next | S]
        C_vals = disc * mean_V
        
        # Store for this step
        policy_data[step] = (edges, C_vals)
        
        # Update V for the next backward step (which is the current step's value)
        # V_curr = max(Payoff(S_curr), C(S_curr))
        # But we need V_curr for *each path* to pass to the previous step.
        # We can get C(S_curr) for each path by interpolating from C_vals using bin_indices.
        C_curr_paths = C_vals[bin_indices]
        
        Payoff_curr = np.maximum(K - S_curr, 0.0)
        V_next = np.maximum(Payoff_curr, C_curr_paths)
        
        # Check time
        if time.time() > deadline - 10.0:
            # If we run out of time, we can stop. The earlier steps will have None.
            # We should handle None in the exercise function.
            break

    class ExercisePolicy:
        def __init__(self, policy_data, n_exercise, K):
            self.policy_data = policy_data
            self.n_exercise = n_exercise
            self.K = K
            self.dt = T / n_exercise

        def exercise(self, current_state, time):
            # time is a scalar or array? "returning a boolean vector" implies current_state is vector.
            # time is likely a scalar corresponding to the current exercise date.
            
            # Determine step index
            # time is in [0, T]. Exercise dates are dt, 2dt, ..., T.
            # If time is 0.0833 (1/12), step is 0.
            # If time is 1.0, step is 11.
            
            if isinstance(time, (int, float, np.integer, np.floating)):
                t_val = float(time)
            else:
                # If time is an array, we might need to handle it, but usually it's a single step call.
                # The contract says "exercise(current_state, time) returning a boolean vector".
                # This implies `time` is a scalar for the batch of paths in `current_state`.
                t_val = float(time[0]) if hasattr(time, '__len__') and len(time) > 0 else 0.0

            # Map time to step index (0-based)
            # step = round(t_val / dt)
            # Clamp to [0, n_exercise - 1]
            step_idx = int(round(t_val / self.dt))
            step_idx = max(0, min(step_idx, self.n_exercise - 1))
            
            data = self.policy_data[step_idx]
            if data is None:
                # Fallback: Exercise if in the money
                return np.maximum(self.K - current_state, 0.0) > 0.0
                
            edges, C_vals = data
            
            # Find bin for each spot in current_state
            # current_state is a 1D array of spots
            bin_indices = np.searchsorted(edges, current_state, side='left')
            bin_indices = np.clip(bin_indices, 0, len(C_vals) - 1)
            
            # Get continuation value for each spot
            C_current = C_vals[bin_indices]
            
            # Calculate payoff
            Payoff = np.maximum(self.K - current_state, 0.0)
            
            # Decision: Exercise if Payoff >= Continuation
            # Add a small epsilon to handle floating point errors?
            # Usually strict inequality is fine, but >= is safer for "exercise if optimal".
            return Payoff >= C_current

    return ExercisePolicy(policy_data, n_exercise, K)

# EVOLVE-BLOCK-END
