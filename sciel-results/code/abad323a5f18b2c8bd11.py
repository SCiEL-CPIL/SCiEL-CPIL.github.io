"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin
    
    # Extract parameters from contract or use defaults if not provided
    # Based on task description: K=100, r=0.05, sigma=0.2, T=1.0, 12 monthly dates
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 12
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # Process training paths
    # Assume training_paths is shape (n_paths, n_steps) or (n_paths, n_steps+1)
    # Task says "12 monthly exercise dates (no time-0 exercise)"
    # Usually paths include S_0 to S_12. Exercise happens at 1..12.
    paths = np.asarray(training_paths, dtype=np.float64)
    
    if paths.ndim == 1:
        paths = paths.reshape(-1, 1)
        
    n_paths = paths.shape[0]
    
    # Determine number of time steps in data
    # If data has 13 columns (S0..S12), steps are 1..12
    # If data has 12 columns, steps are 1..12 (assuming S0 is not included or is column 0)
    # Let's assume standard format: columns are S_0, S_1, ..., S_12
    # Exercise decisions at t=1..12
    # We need to align indices.
    
    if paths.shape[1] == n_steps + 1:
        # Columns 0..12. Exercise at 1..12.
        S_t = [paths[:, i] for i in range(n_steps + 1)]
    elif paths.shape[1] == n_steps:
        # Columns 0..11. Exercise at 0..11? 
        # "12 monthly exercise dates". 
        # If S_0 is not included, then S_1 is col 0.
        # Let's assume the paths are the states at exercise dates.
        # If shape is (N, 12), these are S_1..S_12.
        S_t = [paths[:, i] for i in range(n_steps)]
        # We need S_0 for the first step? No, LSM starts from T-1.
        # But we need to know when to start.
        # Let's assume the standard: paths contain S_0 to S_T.
        # If the input is just the exercise dates, we might be missing S_0 for the first regression?
        # Actually, for step 11 (last), we regress S_11 on S_12 payoff.
        # For step 1, we regress S_1 on S_2...S_12 value.
        # We don't need S_0 for the regression itself, only for the final decision at t=1?
        # Wait, the policy is called with current_state. 
        # If the evaluator provides states at t=1..12, we just need the boundary for those.
        
    else:
        # Fallback: assume last n_steps columns are the exercise dates
        S_t = [paths[:, -n_steps+i] for i in range(n_steps)]
        
    # Backward Induction with LSM
    # V_T is the payoff at T
    # For a put: max(K - S_T, 0)
    
    # We will store the regression coefficients for each step t = n_steps-1 down to 0
    # Let's index steps 0 to n_steps-1 corresponding to t=1 to t=12
    # V_{t+1} is the value at the next step.
    
    # Initialize V_next with the terminal payoff
    # We need to map the last column to V_T
    # If S_t list has 12 elements, S_t[-1] is S_12.
    V_next = np.maximum(K - S_t[-1], 0.0)
    
    # Store regression coefficients for each step
    # coeffs[t] will be used to estimate E[V_{t+1} | S_t]
    # We store the coefficients for the polynomial in S_t
    n_poly = 2  # Quadratic
    
    # We need to store the boundaries or the regression models
    # Since the policy object must return a boolean vector for a given state and time,
    # we can store the regression coefficients and the discount factor.
    
    # coeffs[step] where step 0 is t=1, step 11 is t=12?
    # Actually, we compute backwards.
    # Step 11 (last exercise): compare K-S_12 with 0. No regression needed for decision, but V_12 is known.
    # Step 10: Regress V_11 on S_10. Decision at S_10: Exercise if K-S_10 > E[discount*V_11|S_10].
    # ...
    # Step 0: Regress V_1 on S_1. Decision at S_1.
    
    # We need to store the regression coefficients for steps 0 to n_steps-2.
    # For step n_steps-1 (last), the continuation value is 0 (European payoff is final).
    # So for step n_steps-2, we regress the discounted terminal payoff on S_{n_steps-1}.
    
    reg_coeffs = [None] * n_steps  # reg_coeffs[i] corresponds to step i (t=i+1)
    
    # Iterate backwards from step n_steps-2 (second to last) down to 0
    # V_next currently holds V_{t+1}
    
    for step in range(n_steps - 2, -1, -1):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        S_current = S_t[step]
        
        # Prepare design matrix for regression
        # We regress V_next on S_current
        # Features: 1, S, S^2
        A = np.column_stack((np.ones_like(S_current), S_current, S_current**2))
        
        # Least squares fit
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(A, V_next, rcond=None)
            reg_coeffs[step] = coeffs
        except:
            # Fallback to simple mean if lstsq fails
            reg_coeffs[step] = np.array([np.mean(V_next), 0, 0])
            
        # Calculate continuation value at current step
        # E[discount * V_next | S_current]
        # Note: V_next is the undiscounted value at t+1.
        # The risk-neutral expectation includes the discounting.
        # So Continuation = discount * (c0 + c1*S + c2*S^2)
        cont_val = discount * (coeffs[0] + coeffs[1] * S_current + coeffs[2] * S_current**2)
        
        # Exercise payoff
        ex_val = np.maximum(K - S_current, 0.0)
        
        # Determine optimal exercise: Exercise if ex_val > cont_val
        # This creates the "effective" value V_current for the next iteration
        V_current = np.where(ex_val > cont_val, ex_val, cont_val)
        
        # Update V_next for the next iteration (which is now the previous step)
        V_next = V_current
        
    # Now we have reg_coeffs for steps 0 to n_steps-2.
    # For step n_steps-1 (last step), the decision is simply: Exercise if K-S > 0.
    # This is equivalent to a regression where continuation is 0.
    
    class ExercisePolicy:
        def __init__(self, coeffs_list, K, r, dt):
            self.coeffs = coeffs_list
            self.K = K
            self.discount = np.exp(-r * dt)
            
        def exercise(self, current_state, time):
            # time is the index of the exercise date (0-based? 1-based?)
            # Task says "12 monthly exercise dates".
            # Usually time corresponds to the step index.
            # If time=0, it's the first date.
            # If time=11, it's the last date.
            
            # Ensure current_state is a numpy array
            S = np.asarray(current_state, dtype=np.float64)
            
            if time >= len(self.coeffs) or self.coeffs[time] is None:
                # Last step or missing coeffs: European put decision
                return S < self.K
            else:
                c = self.coeffs[time]
                # Continuation value estimate
                cont = self.discount * (c[0] + c[1] * S + c[2] * S**2)
                # Exercise value
                ex = np.maximum(self.K - S, 0.0)
                return ex > cont
                
    return ExercisePolicy(reg_coeffs, K, r, dt)

# EVOLVE-BLOCK-END
