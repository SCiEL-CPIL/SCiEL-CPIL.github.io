"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option.
    Uses backward induction on a fine grid to compute optimal exercise boundaries.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s margin within 120s limit

    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    # 12 monthly exercise dates (no time-0 exercise)
    dt = T / n_exercises

    # Build a fine grid for backward induction
    # Use a large number of steps for accuracy
    n_steps = 1200
    sub_dt = T / n_steps

    # Grid parameters
    u = np.exp(sigma * np.sqrt(sub_dt))
    d = 1.0 / u
    p = (np.exp(r * sub_dt) - d) / (u - d)
    disc = np.exp(-r * sub_dt)

    # Determine grid bounds for spot values
    # Spot ranges roughly from very low to high
    S_max = 300.0
    S_min = 0.1
    n_grid = 5000
    S_grid = np.linspace(S_min, S_max, n_grid)

    # We'll do backward induction on the grid
    # V represents the continuation value (not including exercise)
    # At each exercise date, we compare exercise value vs continuation value

    # Initialize terminal value (at T, no more exercise, just payoff)
    V = np.maximum(K - S_grid, 0.0)

    # We need to track exercise boundaries at each of the 12 exercise dates
    # Exercise dates are at times: dt, 2*dt, ..., 12*dt (=T)
    # But we exercise at dates 1 through 12 (monthly), so dates at dt, 2dt, ..., 12dt
    # Wait - 12 monthly exercise dates with T=1 means dates at 1/12, 2/12, ..., 12/12
    # The last date is at T=1.0

    # Exercise boundaries: for each exercise date, the maximum S at which we exercise
    # We'll store the boundary (threshold) for each exercise date
    boundaries = np.zeros(n_exercises)

    # Backward induction from last exercise date to first
    # At each exercise date, we compute the continuation value by discounting
    # the expected value from the next step, then compare with exercise value

    # We need to be careful: the exercise dates are at dt, 2dt, ..., 12dt
    # Between exercise dates, there are n_steps/n_exercises sub-steps
    steps_per_exercise = n_steps // n_exercises

    for ex_idx in range(n_exercises - 1, -1, -1):
        # Compute continuation value at this exercise date
        # We need to evolve V forward by steps_per_exercise sub-steps
        # V is the value just after this exercise date (continuation)
        # We compute the expected discounted value from the next state

        # Apply binomial step steps_per_exercise times
        V_cont = V.copy()
        for _ in range(steps_per_exercise):
            V_next = disc * (p * V_cont * u + (1 - p) * V_cont * d)
            # Interpolate to grid
            V_cont = V_next

        # Actually, let me reconsider the binomial tree approach
        # The standard approach: at each node, value = max(exercise, discount * expected_next)
        # For a grid-based approach, we need to handle the spatial evolution

        # Let me use a simpler approach: at each exercise date, compute the continuation
        # value by simulating the risk-neutral dynamics

        # For the grid approach, the continuation value at time t for spot S is:
        # E[exp(-r*dt) * V(t+dt, S')] where S' follows GBM
        # Under risk-neutral: S' = S * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

        # For a put, the continuation value can be computed via the grid
        # Let's use a different approach: solve the PDE or use a fine binomial tree

        # Actually, let's use a proper binomial tree with the grid
        # The issue is that the binomial tree changes the spot, so we need to
        # interpolate V to the new spot values

        # Let me restart with a cleaner approach using a fine binomial tree
        # but only at the exercise dates

    # Let me use a cleaner backward induction approach
    # We'll work with a log-space grid and use the binomial transition

    # Reset
    n_grid = 4000
    S_grid = np.linspace(0.5, 250.0, n_grid)

    # Terminal condition at T
    V = np.maximum(K - S_grid, 0.0)

    # Exercise boundaries for each date (index 0 = first exercise date at dt)
    boundaries = np.zeros(n_exercises)

    # Backward induction
    for ex_idx in range(n_exercises - 1, -1, -1):
        # Compute continuation value at this exercise date
        # We need to evolve V forward by dt (one exercise period)
        # Under risk-neutral GBM: S(t+dt) = S(t) * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

        # For each grid point S_i, the next spot is S_i * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
        # We need E[V(S')]

        # Use a discrete approximation: sample Z at several points
        # Or use the fact that for a binomial tree:
        # S' = S * u with prob p, S' = S * d with prob (1-p)
        # where u = exp(sigma*sqrt(dt)), d = 1/u, p = (exp(r*dt) - d)/(u - d)

        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp(r * dt) - d) / (u - d)
        disc = np.exp(-r * dt)

        # For each S_i, compute S_i * u and S_i * d, then interpolate V
        S_up = S_grid * u
        S_down = S_grid * d

        # Interpolate V at S_up and S_down
        V_up = np.interp(S_up, S_grid, V, left=0.0, right=0.0)
        V_down = np.interp(S_down, S_grid, V, left=0.0, right=0.0)

        # Continuation value
        V_cont = disc * (p * V_up + (1 - p) * V_down)

        # Exercise value
        V_ex = np.maximum(K - S_grid, 0.0)

        # Optimal value: max(exercise, continuation)
        V_opt = np.maximum(V_ex, V_cont)

        # Exercise boundary: maximum S where V_ex >= V_cont
        # Find the largest S where we exercise
        exercise_mask = V_ex >= V_cont
        if np.any(exercise_mask):
            # Find the maximum S where we exercise
            exercise_indices = np.where(exercise_mask)[0]
            max_ex_idx = exercise_indices[-1]
            boundaries[ex_idx] = S_grid[max_ex_idx]
        else:
            boundaries[ex_idx] = 0.0

        # Update V for the next (earlier) exercise date
        V = V_opt

        # Check time
        if time.time() > deadline:
            break

    # Create the exercise policy
    class ExercisePolicy:
        def __init__(self, boundaries, S_grid):
            self.boundaries = boundaries
            self.S_grid = S_grid
            self.n_exercises = len(boundaries)

        def exercise(self, current_state, time):
            """
            current_state: array of spot values
            time: current time (or exercise date index)
            Returns: boolean array indicating whether to exercise
            """
            # Determine which exercise date we're at
            # Exercise dates are at dt, 2dt, ..., 12dt
            # time is the current time
            dt = T / self.n_exercises

            # Find the exercise date index
            # If time is close to k*dt, we're at exercise date k
            # Exercise dates are 1-indexed: 1*dt, 2*dt, ..., 12*dt
            if time <= 0:
                return np.zeros(len(current_state), dtype=bool)

            # Find which exercise period we're in
            # The exercise dates are at dt, 2dt, ..., n_exercises*dt
            # If time is between (k-1)*dt and k*dt, we haven't reached exercise date k yet
            # We exercise at the exercise dates

            # Determine the exercise date index (1 to n_exercises)
            ex_date_idx = int(np.round(time / dt))
            ex_date_idx = max(1, min(ex_date_idx, self.n_exercises))

            # Get the boundary for this exercise date
            # boundaries[0] corresponds to the first exercise date (at dt)
            # boundaries[1] corresponds to the second exercise date (at 2dt)
            # etc.
            boundary_idx = ex_date_idx - 1
            boundary = self.boundaries[boundary_idx]

            # Exercise if spot <= boundary
            return current_state <= boundary

    return ExercisePolicy(boundaries, S_grid)
# EVOLVE-BLOCK-END
