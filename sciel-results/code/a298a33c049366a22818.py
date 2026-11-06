"""Rewritten program for Bermudan exercise-policy research.
Uses a parametric smooth boundary in log-space fitted via grid search on a coarse 1D grid,
decoupling policy representation from fixed-grid backward induction bias.
"""
# EVOLVE-BLOCK-START
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a parametric smooth boundary.
    
    Approach:
    - Represent the exercise boundary $B_i$ at exercise date $i$ as a smooth parametric function.
    - We use a power-law form: $B_i = K \cdot (i/N)^\beta$. This captures the typical concave
      shape of the Bermudan put boundary, rising steeply from 0 and saturating near K.
    - Fit the parameter $\beta$ via grid search on the training paths, using vectorized
      simulation of the exercise policy to maximize the mean discounted payoff.
    """
    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise_dates = contract.get('n_exercise_dates', 12)
    
    # Time step
    dt = T / n_exercise_dates
    
    # Assume training_paths is a 2D array of shape (n_paths, n_exercise_dates + 1)
    # where column 0 is the initial spot and columns 1 to n_exercise_dates are the
    # spot prices at each exercise date.
    paths = np.asarray(training_paths)
    
    if paths.ndim == 1:
        # Single path, reshape
        paths = paths.reshape(1, -1)
    
    n_paths, n_cols = paths.shape
    
    # Extract spot prices at exercise dates (columns 1 to n_exercise_dates)
    S = paths[:, 1:n_exercise_dates + 1]  # shape: (n_paths, n_exercise_dates)
    
    # Discount factors
    discount_factors = np.exp(-r * np.arange(1, n_exercise_dates + 1) * dt)
    
    # Precompute payoff
    payoff = np.maximum(K - S, 0)
    
    # Grid search over beta
    # B_i = K * (i / N)^beta
    # i = 1, ..., N
    i_indices = np.arange(1, n_exercise_dates + 1, dtype=float)
    normalized_i = i_indices / n_exercise_dates
    
    beta_vals = np.linspace(0.1, 3.0, 100)
    
    best_score = -np.inf
    best_beta = 0.5
    
    # Vectorized evaluation for each beta
    for beta in beta_vals:
        boundaries = K * (normalized_i ** beta)
        boundaries = np.clip(boundaries, 0, K)
        
        # Exercise mask: S <= boundaries
        exercise_mask = S <= boundaries[None, :]
        
        # Find first exercise
        # cum_exercise is 0 before first exercise, 1 at and after
        cum_exercise = np.cumsum(exercise_mask, axis=1)
        
        # first_exercise_mask is True only at the first True in each row
        # For j=0, condition is just exercise_mask[:, 0]
        # For j>0, condition is exercise_mask[:, j] AND cum_exercise[:, j-1] == 0
        first_exercise_mask = np.zeros_like(exercise_mask)
        first_exercise_mask[:, 0] = exercise_mask[:, 0]
        for j in range(1, n_exercise_dates):
            first_exercise_mask[:, j] = exercise_mask[:, j] & (cum_exercise[:, j-1] == 0)
            
        # Realized payoff
        realized_payoff = payoff * first_exercise_mask * discount_factors[None, :]
        
        # Mean discounted payoff
        mean_payoff = np.mean(realized_payoff.sum(axis=1))
        
        if mean_payoff > best_score:
            best_score = mean_payoff
            best_beta = beta
    
    # Create the exercise policy
    boundaries = K * (normalized_i ** best_beta)
    boundaries = np.clip(boundaries, 0, K)
    
    class ExercisePolicy:
        def __init__(self, boundaries, n_exercise_dates, dt):
            self.boundaries = boundaries
            self.n_exercise_dates = n_exercise_dates
            self.dt = dt
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise the option.
            
            Parameters:
            - current_state: array of spot prices
            - time: current time (fraction of T) or exercise date index
            
            Returns:
            - boolean vector indicating whether to exercise
            """
            current_state = np.asarray(current_state)
            
            # Determine the exercise date index from time
            # Assume time is the fraction of T (0 to 1)
            # If time > 1, assume it's an index (1 to N)
            if isinstance(time, (int, float)):
                if time <= 1.0:
                    # time is fraction of T
                    # t_i = i * dt, so i = t / dt
                    idx = int(round(time / self.dt))
                else:
                    # time is already the index
                    idx = int(time)
            else:
                # time is an array
                time_arr = np.asarray(time)
                if np.max(time_arr) <= 1.0:
                    idx_arr = np.round(time_arr / self.dt).astype(int)
                else:
                    idx_arr = time_arr.astype(int)
                # If time is an array, we might need to handle vectorized logic,
                # but typically exercise() is called with a scalar time for a batch of states.
                # If time is an array, we assume it matches the shape of current_state?
                # The task says "exercise(current_state, time) returning a boolean vector".
                # Usually time is scalar for a specific step.
                # If time is array, we will try to handle it, but scalar is more likely.
                # For safety, if time is array, we return a boolean array where each element
                # is compared against the boundary for its specific time.
                # But boundaries are indexed by time.
                # Let's assume time is scalar for now as per common interface.
                # If we must support array time:
                # This is complex. Let's stick to scalar time assumption for robustness.
                # If time is array, we'll treat it as a single time step for all? No.
                # Let's assume the caller passes scalar time.
                raise ValueError("Vectorized time not supported, please pass scalar time")
            
            # Clamp index to valid range (1 to N)
            idx = np.clip(idx, 1, self.n_exercise_dates)
            
            # Get the boundary for this exercise date
            # boundaries is 0-indexed array of length N, corresponding to i=1..N
            # So boundary for index i is boundaries[i-1]
            boundary = self.boundaries[idx - 1]
            
            # Exercise if current_state <= boundary
            return current_state <= boundary

    policy = ExercisePolicy(boundaries, n_exercise_dates, dt)
    return policy

# EVOLVE-BLOCK-END
