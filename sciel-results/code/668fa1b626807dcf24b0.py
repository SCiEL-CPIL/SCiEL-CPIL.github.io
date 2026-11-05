"""Rewritten program for Bermudan exercise-policy research."""
# EVOLVE-BLOCK-START
import numpy as np
import math

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on training paths
    and a simple lookup/interpolation scheme for exercise decisions.
    
    The approach:
    1. Parse contract parameters (T, r, sigma, K, number of exercise dates).
    2. Use the training_paths to construct a discrete grid of spot prices at each exercise date.
    3. Perform backward induction to compute optimal continuation values and exercise decisions
       on this grid.
    4. Return an object whose exercise(current_state, time) method uses interpolation on the
       grid to determine whether to exercise.
    
    We use a neural-network-like approach via a simple piecewise-linear decision boundary
    derived from backward induction, which is non-parametric and captures the optimal stopping
    boundary.
    """
    # Extract contract parameters
    T = contract.get('T', 1.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    K = contract.get('K', 100.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Training paths shape: (n_paths, n_time_steps + 1)
    # We assume paths are given at monthly intervals including t=0
    n_paths, n_steps = training_paths.shape
    dt = T / (n_steps - 1)
    
    # The exercise dates are the last n_exercise time steps (excluding t=0)
    # So exercise dates are at indices: n_steps - n_exercise, ..., n_steps - 1
    # But we need to be careful: 12 monthly exercise dates, no time-0 exercise.
    # If n_steps = 13 (t=0 to t=12 months), exercise dates are indices 1..12.
    
    # Determine which time indices correspond to exercise dates
    # Typically, exercise dates are at the end of each month, so indices 1 to n_exercise
    # But if n_steps > n_exercise + 1, we need to map correctly.
    
    # Let's assume the paths are at monthly intervals and exercise dates are the last
    # n_exercise time points (indices n_steps - n_exercise to n_steps - 1).
    # For 12 monthly dates and T=1, n_steps should be 13 (t=0, 1, ..., 12).
    
    # Build a price grid for each exercise date
    # Use quantiles of the training paths to create a reasonable grid
    n_grid = 200  # number of grid points per exercise date
    
    # Collect prices at each exercise date
    exercise_indices = list(range(n_steps - n_exercise, n_steps))
    
    # Build grid for each exercise date
    grids = {}
    for idx in exercise_indices:
        prices = training_paths[:, idx]
        # Use quantile-based grid
        qs = np.linspace(0.01, 0.99, n_grid)
        grid = np.quantile(prices, qs)
        grids[idx] = grid
    
    # Backward induction: compute continuation values and exercise decisions
    # Start from the last exercise date
    # At the last exercise date, the value is max(K - S, 0)
    
    # Store continuation values and exercise decisions for each exercise date
    cont_values = {}
    exercise_decisions = {}
    
    # Last exercise date
    last_idx = exercise_indices[-1]
    grid_last = grids[last_idx]
    payoff_last = np.maximum(K - grid_last, 0.0)
    cont_values[last_idx] = payoff_last
    exercise_decisions[last_idx] = np.ones(n_grid, dtype=bool)  # Always exercise at last date for a put if ITM
    
    # Work backwards
    discount_factor = math.exp(-r * dt)
    
    for i in range(len(exercise_indices) - 2, -1, -1):
        current_idx = exercise_indices[i]
        next_idx = exercise_indices[i + 1]
        
        grid_current = grids[current_idx]
        grid_next = grids[next_idx]
        cont_next = cont_values[next_idx]
        
        # For each price in the current grid, estimate the expected discounted continuation value
        # We need to simulate the transition from current price to next price under risk-neutral GBM
        # Under risk-neutral measure: S_{t+dt} = S_t * exp((r - sigma^2/2) * dt + sigma * sqrt(dt) * Z)
        
        # For a given current price S, the next price is lognormal:
        # S_next = S * exp((r - sigma^2/2) * dt + sigma * sqrt(dt) * Z)
        # where Z ~ N(0, 1)
        
        # We can compute the expected continuation value by integrating over Z
        # E[cont(S_next)] = integral cont(S * exp(mu + sigma*sqrt(dt)*Z)) phi(Z) dZ
        # where mu = (r - sigma^2/2) * dt
        
        mu = (r - sigma**2 / 2) * dt
        sigma_dt = sigma * math.sqrt(dt)
        
        # Use quadrature with Gauss-Hermite or simple Monte Carlo
        # For speed, use a fixed set of quadrature points
        n_quad = 50
        z_points = np.linspace(-4, 4, n_quad)
        weights = np.ones(n_quad) / n_quad  # Simple uniform weights (crude but fast)
        
        # More accurate: use Gauss-Hermite quadrature
        # But for simplicity and speed, use a reasonable number of points
        
        # Actually, let's use a better quadrature: trapezoidal on a fine grid
        n_quad = 100
        z_points = np.linspace(-4, 4, n_quad)
        dz = z_points[1] - z_points[0]
        weights = np.ones(n_quad) * dz
        
        # Compute the expected continuation value for each current price
        # S_next = S * exp(mu + sigma_dt * Z)
        # We need to interpolate cont_next at S_next
        
        # Precompute the log of grid_next for interpolation
        log_grid_next = np.log(grid_next)
        
        expected_cont = np.zeros(n_grid)
        for j in range(n_grid):
            S_j = grid_current[j]
            if S_j <= 0:
                expected_cont[j] = 0
                continue
            # S_next = S_j * exp(mu + sigma_dt * Z)
            log_S_next = math.log(S_j) + mu + sigma_dt * z_points
            S_next = np.exp(log_S_next)
            
            # Interpolate cont_next at S_next
            # Use np.interp
            cont_at_next = np.interp(S_next, grid_next, cont_next)
            expected_cont[j] = np.sum(weights * cont_at_next)
        
        expected_cont *= discount_factor
        
        # Exercise payoff at current date
        payoff_current = np.maximum(K - grid_current, 0.0)
        
        # Optimal value is max(payoff, expected continuation)
        optimal_value = np.maximum(payoff_current, expected_cont)
        
        # Exercise decision: exercise if payoff >= expected continuation
        exercise_decisions[current_idx] = payoff_current >= expected_cont - 1e-10
        
        cont_values[current_idx] = optimal_value
    
    # Now create the policy object
    class ExercisePolicy:
        def __init__(self, grids, exercise_decisions, exercise_indices, n_exercise, T, dt):
            self.grids = grids
            self.exercise_decisions = exercise_decisions
            self.exercise_indices = exercise_indices
            self.n_exercise = n_exercise
            self.T = T
            self.dt = dt
            
        def exercise(self, current_state, time):
            """
            current_state: array of current spot prices (shape: (n_eval_paths,))
            time: current time (float)
            
            Returns: boolean vector indicating whether to exercise each path.
            """
            # Determine which exercise date this time corresponds to
            # Exercise dates are at times: (n_steps - n_exercise + 1) * dt, ..., n_steps * dt
            # Or more simply: the exercise dates are at t = dt, 2dt, ..., n_exercise * dt
            # if n_steps = n_exercise + 1
            
            # Find the closest exercise index
            # time is in [0, T], and exercise dates are at specific times
            
            # The exercise dates in time units:
            # If n_steps = 13, dt = 1/12, exercise dates at t = 1/12, 2/12, ..., 12/12
            # In general, exercise date k (0-indexed) is at time (n_steps - n_exercise + 1 + k) * dt
            # But this is getting complicated. Let's just find the closest exercise index.
            
            # Compute the expected time for each exercise index
            # exercise_indices are array indices, time = index * dt
            # So for a given time, find the exercise index whose index * dt is closest to time
            
            idx_times = np.array([idx * self.dt for idx in self.exercise_indices])
            closest_idx = np.argmin(np.abs(idx_times - time))
            ex_idx = self.exercise_indices[closest_idx]
            
            grid = self.grids[ex_idx]
            decisions = self.exercise_decisions[ex_idx]
            
            # For each current_state, find the closest grid point and use the decision
            # Use np.searchsorted for efficient lookup
            # We need to map each current_state to the nearest grid point
            
            # Use interpolation: for each current_state, find the two nearest grid points
            # and use the decision of the nearest one (or interpolate if we had continuous decisions)
            # Since decisions are binary, just use nearest neighbor.
            
            # Efficiently find nearest grid point for each state
            # Use searchsorted
            insert_pos = np.searchsorted(grid, current_state)
            
            # Handle boundaries
            insert_pos = np.clip(insert_pos, 1, len(grid) - 1)
            
            # Compare with left and right neighbors
            left_idx = insert_pos - 1
            right_idx = insert_pos
            
            dist_left = np.abs(current_state - grid[left_idx])
            dist_right = np.abs(current_state - grid[right_idx])
            
            nearest = np.where(dist_left <= dist_right, left_idx, right_idx)
            
            # Clip to valid range
            nearest = np.clip(nearest, 0, len(grid) - 1)
            
            return decisions[nearest]
    
    return ExercisePolicy(grids, exercise_decisions, exercise_indices, n_exercise, T, dt)
# EVOLVE-BLOCK-END
