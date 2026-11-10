"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option.
    Uses a 1200-step binomial tree to determine optimal exercise boundaries.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, leave margin
    
    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    # 12 monthly exercise dates
    dt = T / n_exercises
    
    # Build a 1200-step binomial tree for accurate boundary estimation
    # But we need to be careful about time budget
    # Let's use a moderate number of steps for the tree
    n_steps = 1200
    
    # Binomial tree parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Create price grid
    # Start from current spot (use 100 as reference for tree construction)
    S0 = 100.0
    
    # Build the tree
    # At each exercise date, we need to decide whether to exercise
    # We'll use dynamic programming from the last exercise date backwards
    
    # Number of nodes at each level
    # We'll use a simplified approach: create a grid of prices
    # and compute the optimal exercise boundary at each exercise date
    
    # For a put option, it's optimal to exercise when S <= boundary(t)
    # We need to find these boundaries
    
    # Use a fine grid for price levels
    # Range of prices: from very low to high
    S_min = 1.0
    S_max = 200.0
    n_price_levels = 1000
    
    S_grid = np.linspace(S_min, S_max, n_price_levels)
    
    # Discount factor per period
    df = np.exp(-r * dt)
    
    # Initialize value at final exercise date (European put)
    # At the last exercise date, value = max(K - S, 0)
    value_next = np.maximum(K - S_grid, 0.0)
    
    # Store boundaries for each exercise date
    boundaries = [None] * n_exercises
    
    # Work backwards from last exercise date to first
    for t in range(n_exercises - 1, -1, -1):
        if time.time() > deadline - 5.0:
            # Not enough time, break and use what we have
            break
        
        # Compute continuation value
        # Expected value at next step
        S_up = S_grid * u
        S_down = S_grid * d
        
        # Interpolate value_next at S_up and S_down
        # S_up and S_down may be outside the grid range
        # Use np.interp which handles out-of-bounds by using endpoint values
        val_up = np.interp(S_up, S_grid, value_next)
        val_down = np.interp(S_down, S_grid, value_next)
        
        cont_value = df * (p * val_up + (1 - p) * val_down)
        
        # Exercise value
        ex_value = np.maximum(K - S_grid, 0.0)
        
        # Optimal value
        value_current = np.maximum(ex_value, cont_value)
        
        # Find exercise boundary: largest S where exercise is optimal
        # Exercise when ex_value >= cont_value
        exercise_mask = ex_value >= cont_value
        if np.any(exercise_mask):
            boundary = np.max(S_grid[exercise_mask])
        else:
            boundary = 0.0  # Never exercise
        
        boundaries[t] = boundary
        
        value_next = value_current
    
    # Create the exercise policy
    class ExercisePolicy:
        def __init__(self, boundaries, K, r, dt):
            self.boundaries = boundaries
            self.K = K
            self.r = r
            self.dt = dt
        
        def exercise(self, current_state, time):
            """
            current_state: array of current stock prices
            time: current time (continuous)
            Returns: boolean array indicating whether to exercise
            """
            # Determine which exercise period we're in
            # Exercise dates are at t = dt, 2*dt, ..., 12*dt
            # time is the current time
            
            # Find the exercise index
            # If time is close to an exercise date, use that boundary
            # We'll round to the nearest exercise date
            
            # Number of periods elapsed
            period = np.round(time / self.dt)
            
            # Clamp to valid range [0, 11]
            period = np.clip(period, 0, len(self.boundaries) - 1)
            
            # Get the boundary for this period
            # Since period may vary per path (if time is array), 
            # but time is typically a scalar in this context
            if isinstance(time, np.ndarray):
                # Handle array time
                period = np.round(time / self.dt)
                period = np.clip(period, 0, len(self.boundaries) - 1)
                # Get boundaries for each path
                bounds = [self.boundaries[int(p)] for p in period]
                bounds = np.array(bounds)
            else:
                bounds = self.boundaries[int(period)]
            
            # Exercise if current_state <= boundary
            # For puts, exercise when S <= boundary
            if isinstance(bounds, np.ndarray):
                return current_state <= bounds
            else:
                return current_state <= bounds
    
    return ExercisePolicy(boundaries, K, r, dt)
# EVOLVE-BLOCK-END
