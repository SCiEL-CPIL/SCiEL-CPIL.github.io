import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (limit is 150s)
    
    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    # Determine number of steps
    N, T_steps = training_paths.shape
    
    # The contract states: "12 monthly exercise dates (no time-0 exercise)"
    # Standard GBM paths usually include t=0.
    # If T_steps == 13, indices are 0..12. Exercise at 1..12.
    # If T_steps == 12, indices are 0..11. This is ambiguous. 
    # However, "12 monthly exercise dates" implies 12 decision points.
    # Let's assume the standard case where the path includes the initial spot.
    # So if T_steps is 13, we have 12 intervals.
    # If T_steps is 12, maybe it's just the exercise dates?
    # Let's check the shape.
    # Usually, training_paths is (N, num_steps).
    # If num_steps = 13, dt = T/12.
    # If num_steps = 12, dt = T/11? Or is t=0 missing?
    # Given "no time-0 exercise", the first exercise is at t=1.
    # If the array has 12 columns, and they are exercise dates, then t=1..12.
    # But GBM paths usually start at t=0.
    # Let's assume the most standard convention: T_steps is the number of time points.
    # If T_steps == 13, then t=0, 1, ..., 12.
    # If T_steps == 12, it's likely t=1, ..., 12 (excluding t=0) or t=0, ..., 11.
    # However, the prompt says "Spot 90/100/110... T 1".
    # Let's look at the "12 monthly exercise dates".
    # This strongly suggests 12 decision points.
    # Let's infer dt.
    # If T_steps == 13: dt = T / 12.
    # If T_steps == 12: It's risky. Let's assume the input includes t=0.
    # Most likely T_steps is 13.
    
    n_exercises = 12
    # We will assume the last column is the final time T.
    # And we exercise at the last n_exercises columns.
    # If T_steps is 13, we exercise at indices 1..12.
    # If T_steps is 12, we exercise at indices 0..11? No, "no time-0 exercise".
    # So if T_steps is 12, it must be indices 1..12? But there are only 12 columns.
    # This implies if T_steps is 12, the first column is t=1?
    # Let's assume T_steps is 13 for safety, or handle dynamically.
    
    # Let's define the exercise indices.
    # We want the last 12 columns to be the exercise dates.
    # If T_steps == 13: exercise at 1, 2, ..., 12.
    # If T_steps == 12: exercise at 0, 1, ..., 11? But "no time-0 exercise".
    # This suggests that if T_steps == 12, the time-0 is NOT included, so index 0 is t=1.
    # Let's check the dt calculation.
    # If T_steps == 13, dt = T/12.
    # If T_steps == 12, dt = T/12 as well (assuming 12 steps).
    
    # Let's assume the number of intervals is 12.
    # So dt = T / 12.
    # The exercise dates are the last 12 points in the path.
    # If T_steps == 13, indices 1..12.
    # If T_steps == 12, indices 0..11 (but these correspond to t=1..12).
    
    # To be robust:
    # Let's find the start index for the first exercise.
    # If T_steps == 13, start_idx = 1.
    # If T_steps == 12, start_idx = 0.
    # In general, start_idx = T_steps - n_exercises.
    
    start_idx = T_steps - n_exercises
    
    if start_idx < 0:
        start_idx = 0 # Fallback
    
    dt = T / n_exercises
    discount = np.exp(-r * dt)
    
    # betas[i] corresponds to the exercise date at index (start_idx + i)
    # i ranges from 0 to 11.
    # i=0 corresponds to the first exercise date.
    # i=11 corresponds to the last exercise date.
    
    betas = [None] * n_exercises
    
    # Start from the last exercise date
    # Last exercise index in the array: T_steps - 1
    last_idx = T_steps - 1
    
    # Value at the last step (intrinsic value)
    S_last = training_paths[:, last_idx]
    V_next = np.maximum(K - S_last, 0.0)
    
    # Loop backward from the second-to-last exercise date to the first.
    # We need to fit a regression for each exercise date to predict the continuation value.
    # The continuation value at time t is the discounted expected value at t+1.
    # We regress V_next (which is the value at t+1) on the state at t.
    
    # The exercise dates are at indices: start_idx, start_idx+1, ..., T_steps-1.
    # Let k be the index in the betas array (0..11).
    # The array index for this exercise date is: arr_idx = start_idx + k.
    # The next array index is: next_arr_idx = arr_idx + 1.
    
    # We iterate k from 11 down to 0.
    # For k=11 (last exercise), we don't need a regression for the decision (always exercise if ITM),
    # but we DO need the value V_next for the previous step (k=10).
    # Wait, the loop structure:
    # 1. V_next is initialized to intrinsic value at the last step.
    # 2. For k = 10 down to 0:
    #    - Current state is at arr_idx = start_idx + k.
    #    - Next state value is V_next (which was computed for arr_idx + 1).
    #    - We regress V_next on S_curr to get the continuation value at arr_idx.
    #    - We compute V_curr = max(Intrinsic, Continuation).
    #    - Set V_next = V_curr for the next iteration (k-1).
    
    # So we loop k from 10 down to 0.
    # What about k=11?
    # At k=11, the value is just intrinsic. No regression needed for the decision.
    # But we need to store the beta for k=11? No, because at maturity, the policy is deterministic: exercise if ITM.
    # However, if we want to use the same logic, we could fit a regression, but it's unnecessary.
    # Let's just handle k=11 specially in the policy.
    
    # Loop for k = 10 down to 0
    for k in range(10, -1, -1):
        if time.time() > deadline:
            break
            
        arr_idx = start_idx + k
        S_curr = training_paths[:, arr_idx]
        
        # Features for regression
        # Use polynomial features: 1, S, S^2, S^3
        # This is a simple approach. More features could help, but might overfit.
        # Let's stick to cubic polynomial.
        X = np.column_stack([
            np.ones_like(S_curr),
            S_curr,
            S_curr**2,
            S_curr**3
        ])
        
        # Target is discounted next value
        # V_next contains the values at arr_idx + 1.
        Y = discount * V_next
        
        # Fit linear regression
        try:
            beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        except:
            beta = np.zeros(X.shape[1])
            
        betas[k] = beta
        
        # Calculate continuation value for current step
        V_cont = X @ beta
        
        # Intrinsic value at current step
        V_intrinsic = np.maximum(K - S_curr, 0.0)
        
        # Optimal value
        V_next = np.maximum(V_intrinsic, V_cont)
        
    # Create policy object
    class Policy:
        def __init__(self, betas, K, r, dt, start_idx):
            self.betas = betas
            self.K = K
            self.r = r
            self.dt = dt
            self.start_idx = start_idx
            self.discount = np.exp(-r * dt)
            
        def exercise(self, current_state, time):
            # time: integer 1..12
            # k = time - 1 (0-based index in betas)
            # arr_idx = start_idx + k
            
            S = np.asarray(current_state)
            intrinsic = np.maximum(self.K - S, 0.0)
            
            if time < 1 or time > 12:
                return intrinsic > 0.0
            
            k = time - 1
            
            # If it's the last exercise date (k=11), always exercise if ITM
            if k == 11:
                return intrinsic > 0.0
            
            beta = self.betas[k]
            if beta is None:
                return intrinsic > 0.0
            
            # Compute features
            X = np.column_stack([
                np.ones_like(S),
                S,
                S**2,
                S**3
            ])
            
            V_cont = X @ beta
            
            return intrinsic > V_cont

    return Policy(betas, K, r, dt, start_idx)
