"""Bermudan exercise-policy: Longstaff-Schwartz regression-based approach."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    # Parse contract parameters
    S0_list = contract.get('S0_list', [90.0, 100.0, 110.0])
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Exercise times: T/12, 2T/12, ..., T (indices 1 to n_dates)
    dt = T / n_dates
    exercise_times = np.array([i * dt for i in range(1, n_dates + 1)])
    
    # Generate training paths if not provided or to ensure consistency
    # The prompt says "4096 training paths". We assume training_paths is provided 
    # in the format expected by the evaluator. 
    # Usually, training_paths is an array of shape (n_paths, n_dates+1) or similar.
    # Let's assume training_paths is (n_paths, n_steps) where n_steps = n_dates + 1 (including t=0).
    # If training_paths is None or empty, we might need to generate, but the contract 
    # implies we receive them. The seed program passed them to stopping_baseline.
    
    if training_paths is None or len(training_paths) == 0:
        # Fallback: Generate paths if not provided (unlikely in this suite)
        n_paths = 4096
        z = np.random.randn(n_paths, n_dates)
        S = np.zeros((n_paths, n_dates + 1))
        S[:, 0] = 100.0 # Use a representative S0 or average
        for i in range(1, n_dates + 1):
            S[:, i] = S[:, i-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z[:, i-1])
        training_paths = S
    else:
        # Ensure it's a numpy array
        training_paths = np.asarray(training_paths, dtype=float)
        
    n_paths, n_steps = training_paths.shape
    # n_steps should be n_dates + 1 (0 to n_dates)
    # Exercise dates are indices 1 to n_dates in the path array (corresponding to times exercise_times)
    
    # Pre-compute discount factors
    # Discount from time t_i to t_{i-1} is exp(-r * dt)
    disc = np.exp(-r * dt)
    
    # We need to store the regression coefficients for each exercise date.
    # Coeffs[i] corresponds to the regression fit at exercise date i (1-based in time, 
    # but we store in a list indexed by step index 1..n_dates).
    # Let's use a list of coefficient vectors.
    coeffs = [None] * (n_dates + 1) # coeffs[i] for exercise date i
    
    # Basis function: Polynomial in log(S)
    # Degree 3 is usually sufficient for puts/calls in 1D.
    deg = 3
    
    def build_basis(S_vals):
        """
        Build the design matrix for regression.
        S_vals: array of shape (n_paths,)
        Returns: array of shape (n_paths, deg+1)
        """
        log_S = np.log(S_vals)
        # Center and scale might help, but simple powers work if range is bounded.
        # S is around 50-150. log(S) is around 3.9-5.1.
        # Let's use raw powers for simplicity and stability with lstsq.
        X = np.ones((len(S_vals), deg + 1))
        for k in range(1, deg + 1):
            X[:, k] = log_S ** k
        return X
    
    # Backward Induction
    # Start from the last exercise date (n_dates)
    # At the last date, continuation value is 0 (European payoff at maturity if not exercised earlier? 
    # No, Bermudan: if not exercised at last date, it's worthless? Or is it a European option at T?
    # "Bermudan put ... 12 monthly exercise dates (no time-0 exercise)".
    # Usually, if you don't exercise at the last date, the option expires worthless.
    # So, at step n_dates, Continuation Value = 0.
    # Payoff = max(K - S, 0).
    # Decision: Exercise if Payoff > 0. (Always exercise if in the money at last step).
    
    # For i from n_dates-1 down to 1:
    # 1. Determine which paths were NOT exercised at step i+1.
    #    Wait, the standard LS algorithm works on the "continuation value" of the *surviving* paths.
    #    Actually, the standard implementation is:
    #    V_{i+1} is the value vector from the next step (discounted back to i).
    #    We regress V_{i+1} on S_i for all paths (or just those not exercised? 
    #    Standard LS uses all paths to fit the continuation value, but the decision is made 
    #    by comparing Payoff_i with fitted V_{i+1}.
    #    However, to avoid bias, some variants use only non-exercised paths for regression.
    #    Let's stick to the standard: Regress the *discounted* value of the next step 
    #    (which already incorporates the optimal decision at i+1) on the current state S_i.
    
    # Initialize V_next (value at step i+1, discounted to i? No, let's keep it as value at time t_{i+1})
    # Let V[i] be the value of the option at time t_i, *before* the decision at t_i?
    # No, let V[i] be the *continuation value* at t_i, i.e., the value if we do NOT exercise at t_i.
    # Then Decision at t_i: Exercise if Payoff_i > V[i].
    # Value at t_i: V_opt[i] = max(Payoff_i, V[i]).
    
    # Step n_dates (last):
    # Payoff_last = max(K - S[:, n_dates], 0)
    # Continuation Value at n_dates = 0 (option expires)
    # So V_opt[n_dates] = Payoff_last.
    # We need to regress the *discounted* V_opt[n_dates] to get the continuation value at n_dates-1.
    
    # Let's define:
    # V_cont[i]: Estimated continuation value at time t_i (value of holding, not exercising).
    # V_opt[i]: Optimal value at time t_i (max(Payoff, V_cont)).
    
    # At i = n_dates:
    # V_cont[n_dates] = 0
    # V_opt[n_dates] = max(K - S[:, n_dates], 0)
    
    # For i = n_dates - 1 down to 1:
    # Target for regression at step i is: disc * V_opt[i+1]
    # We regress this target on S[:, i] to get V_cont[i].
    # Then V_opt[i] = max(K - S[:, i], V_cont[i]).
    
    # Implementation:
    # We need to store the coefficients for V_cont[i] so we can evaluate it later in exercise().
    
    # Last step
    S_last = training_paths[:, n_dates]
    V_opt_next = np.maximum(K - S_last, 0.0)
    
    # Iterate backwards
    for i in range(n_dates - 1, 0, -1):
        if time.time() > deadline - 5.0:
            # Break if running out of time, keep last computed coeffs
            break
            
        S_curr = training_paths[:, i]
        
        # Target: Discounted optimal value from next step
        target = disc * V_opt_next
        
        # Fit regression: target = X @ coeffs
        X = build_basis(S_curr)
        
        # Use least squares
        # lstsq returns (coeffs, residuals, rank, sv)
        coeffs_i, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
        coeffs[i] = coeffs_i
        
        # Predicted continuation value
        V_cont_curr = X @ coeffs_i
        
        # Immediate payoff
        payoff_curr = np.maximum(K - S_curr, 0.0)
        
        # Optimal value
        V_opt_curr = np.maximum(payoff_curr, V_cont_curr)
        
        # Update V_opt_next for the next iteration (which is i-1)
        V_opt_next = V_opt_curr
        
    # If we broke out early, coeffs for earlier steps might be None. 
    # We should handle this in the exercise function or ensure we have at least the last few.
    # For safety, if coeffs[i] is None, we can approximate with the previous one or a simple heuristic.
    # But usually, 12 steps with 4096 paths is very fast (< 1s).
    
    class BermudanExercisePolicy:
        def __init__(self, coeffs, K, r, sigma, T, n_dates, S0_list, contract_params):
            self.coeffs = coeffs
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_dates = n_dates
            self.dt = T / n_dates
            self.exercise_times = np.array([i * self.dt for i in range(1, self.n_dates + 1)])
            self.S0_list = S0_list
            
        def _get_coefficients_for_time(self, t):
            # Find the exercise index corresponding to time t
            # t is in [0, T]. Exercise dates are at dt, 2dt, ..., T.
            # If t is between dates, we use the coefficients for the *current* decision point.
            # The exercise() function is called with current_state and time.
            # The decision is made at the specific exercise date.
            # If t is not exactly an exercise date, we should probably snap to the nearest 
            # upcoming or current exercise date. 
            # The contract says "12 monthly exercise dates".
            # Let's find the index k such that exercise_times[k-1] <= t <= exercise_times[k]
            # Actually, exercise is only allowed AT the dates.
            # If called with a time not exactly a date, we assume it's the closest date 
            # or the one that is currently active.
            # Given the discrete nature, t should be one of the exercise times.
            
            # Find the index i (1-based) such that exercise_times[i-1] is close to t.
            # If t is very close to 0, no exercise. But t=0 is not an exercise date.
            
            # Map t to index 1..n_dates
            # idx = round(t / dt)
            idx = int(np.round(t / self.dt))
            if idx < 1:
                return None
            if idx > self.n_dates:
                return self.coeffs[self.n_dates] # Use last known if out of bounds
            
            return self.coeffs[idx]

        def exercise(self, current_state, time):
            """
            current_state: float or array of floats (spot prices)
            time: float
            Returns: boolean vector (same shape as current_state)
            """
            S = np.asarray(current_state, dtype=float)
            scalar_input = S.ndim == 0
            if scalar_input:
                S = S.reshape(1)
            
            # Get coefficients for this time step
            c = self._get_coefficients_for_time(time)
            
            if c is None:
                # No exercise possible (t=0 or before first date)
                result = np.zeros(len(S), dtype=bool)
                return result[0] if scalar_input else result
            
            # Build basis for current states
            log_S = np.log(S)
            deg = 3
            X = np.ones((len(S), deg + 1))
            for k in range(1, deg + 1):
                X[:, k] = log_S ** k
            
            # Continuation value
            V_cont = X @ c
            
            # Payoff
            payoff = np.maximum(self.K - S, 0.0)
            
            # Exercise if payoff > continuation value
            result = payoff > V_cont
            
            return result[0] if scalar_input else result

    return BermudanExercisePolicy(coeffs, K, r, sigma, T, n_dates, S0_list, contract)
# EVOLVE-BLOCK-END
