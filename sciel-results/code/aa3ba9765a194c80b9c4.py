"""Bermudan exercise-policy research: High-resolution Binomial Tree."""
# EVOLVE-BLOCK-START
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a High-Resolution Binomial Tree.
    
    Parameters:
    -----------
    training_paths : ndarray
        Training price paths (ignored for computation).
    contract : dict or object
        Contract specification with fields like K, r, sigma, T, n_steps, etc.
    
    Returns:
    --------
    object with exercise(current_state, time) method returning boolean vector.
    """
    # Extract contract parameters
    K = getattr(contract, 'K', 100.0)
    r = getattr(contract, 'r', 0.05)
    sigma = getattr(contract, 'sigma', 0.2)
    T = getattr(contract, 'T', 1.0)
    n_exercise_dates = getattr(contract, 'n_steps', 12)
    
    # Ensure numeric types
    K = float(K)
    r = float(r)
    sigma = float(sigma)
    T = float(T)
    n_exercise_dates = int(n_exercise_dates)
    
    # If n_exercise_dates is not provided or is 0, try to infer from training_paths
    if n_exercise_dates <= 0:
        if isinstance(training_paths, np.ndarray):
            n_exercise_dates = training_paths.shape[1] - 1
        else:
            n_exercise_dates = 12
            
    # High-resolution tree parameters
    # Use 100 steps per exercise period for high accuracy
    steps_per_period = 100
    N = n_exercise_dates * steps_per_period
    dt = T / N
    
    # Binomial Tree Parameters (CRR)
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # S0 for tree construction. The boundary is an absolute price level.
    # We set S0 = 1.0. The resulting boundaries are in the same units as K.
    S0 = 1.0
    
    # Initialize values at maturity (step N)
    js_N = np.arange(N + 1)
    prices_next = S0 * (u ** js_N) * (d ** (N - js_N))
    vals_next = np.maximum(K - prices_next, 0.0)
    
    # Boundaries for the n_exercise_dates
    boundaries = np.zeros(n_exercise_dates)
    
    # Boundary at maturity (step N, date index n_exercise_dates - 1)
    boundaries[n_exercise_dates - 1] = K
    
    # Iterate from step N-1 down to 1
    for step_next in range(N, 0, -1):
        step_curr = step_next - 1
        
        # Compute prices for step_curr
        js_curr = np.arange(step_curr + 1)
        prices_curr = S0 * (u ** js_curr) * (d ** (step_curr - js_curr))
        
        # Compute continuation values for step_curr
        # cont[j] = discount * (p * vals_next[j+1] + (1-p) * vals_next[j])
        exp_vals = p * vals_next[1:] + (1 - p) * vals_next[:-1]
        cont_vals = discount * exp_vals
        
        # Immediate payoff
        imm_payoff = np.maximum(K - prices_curr, 0.0)
        
        # Optimal value
        vals_curr = np.maximum(imm_payoff, cont_vals)
        
        # Check if step_curr is an exercise date
        if step_curr % steps_per_period == 0:
            exercise_mask = imm_payoff >= cont_vals
            
            if np.any(exercise_mask):
                idxs = np.where(exercise_mask)[0]
                j_star = idxs[-1]
                boundary_val = prices_curr[j_star]
                
                k = step_curr // steps_per_period
                date_idx = k - 1
                
                if 0 <= date_idx < n_exercise_dates:
                    boundaries[date_idx] = boundary_val
            else:
                k = step_curr // steps_per_period
                date_idx = k - 1
                if 0 <= date_idx < n_exercise_dates:
                    boundaries[date_idx] = 0.0
        
        # Update for next iteration
        vals_next = vals_curr
        
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, n_steps, dt, n_exercise_dates):
            self.boundaries = boundaries
            self.n_steps = n_steps
            self.dt = dt
            self.n_exercise_dates = n_exercise_dates
            
        def exercise(self, current_state, time):
            """
            Determine whether to exercise.
            
            Parameters:
            -----------
            current_state : ndarray of shape (n_eval_paths,)
                Current prices.
            time : int or float
                Time index or time value.
            
            Returns:
            --------
            ndarray of bools, shape (n_eval_paths,)
            """
            # Convert time to exercise date index (0-based)
            if isinstance(time, (int, np.integer)):
                idx = int(time) - 1
            else:
                T_total = self.dt * self.n_steps
                dt_exercise = T_total / self.n_exercise_dates
                k = int(round(float(time) / dt_exercise))
                idx = k - 1
            
            # Clamp index
            idx = max(0, min(idx, self.n_exercise_dates - 1))
            
            boundary = self.boundaries[idx]
            return np.array(current_state) <= boundary

    return ExercisePolicy(boundaries, N, dt, n_exercise_dates)
# EVOLVE-BLOCK-END
