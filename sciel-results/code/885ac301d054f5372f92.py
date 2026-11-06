"""Bermudan exercise-policy: Regression-based backward induction with training paths."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Parse contract parameters
    S0_list = contract.get('S0_list', [90.0, 100.0, 110.0])
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)

    # Exercise times: T/12, 2T/12, ..., T (no time-0 exercise)
    dt = T / n_dates
    exercise_times = np.array([(i + 1) * dt for i in range(n_dates)])

    # training_paths shape: (n_paths, n_dates) - spot prices at each exercise date
    n_paths, n_obs = training_paths.shape

    # We'll do backward induction using regression to estimate continuation values
    # For each exercise date j (from last to first), we fit a function V_j(S) that
    # gives the continuation value (value of holding) as a function of spot S.
    # The exercise policy at date j is: exercise if (K - S) > V_j(S)

    # Discount factors
    discount = np.exp(-r * dt)

    # Payoff at maturity (last exercise date, index n_dates-1)
    # At the last date, there's no continuation, so we always exercise for a put if K > S
    # But wait - at the last date, we compare exercise payoff to 0 (no further dates)
    # So at last date: exercise if K - S > 0, i.e., S < K

    # We need to estimate V_j(S) for j = 0, ..., n_dates-2
    # V_j(S) = E[discount * max(exercise at j+1, continue)] | S_j = S

    # Strategy: Use the training paths to do backward induction with regression.
    # At each step j, given the simulated paths, we know S_j and S_{j+1}.
    # We can compute the optimal action at j+1 (using V_{j+1} if j+1 < n_dates-1, 
    # or just the payoff if j+1 = n_dates-1).
    # Then the value at j is: max(K - S_j, discount * E[V_{j+1}(S_{j+1}) | S_j])
    # We fit a regression of the continuation value on S_j.

    # For regression, we'll use piecewise linear or polynomial fits on S.
    # Let's use a grid-based approach: bin S values and compute average continuation.

    # Actually, let's use a more principled approach:
    # 1. At the last date (j = n_dates-1), the value is max(K - S, 0) - always exercise if in the money
    # 2. Working backwards, at date j, for each path we compute:
    #    - Exercise payoff: max(K - S_j, 0)
    #    - Continuation value: discount * V_{j+1}(S_{j+1})
    #    - The optimal value: max(exercise, continuation)
    # 3. We need to estimate V_j(S) = E[max(exercise at j, discount * V_{j+1}(S_{j+1})) | S_j = S]
    #    This is tricky because the optimal action depends on S_j and the distribution of S_{j+1}|S_j.

    # Alternative: Direct simulation-based backward induction
    # At each step, for each path, we know S_j and S_{j+1}.
    # We can compute the "implied" continuation value for each path at step j as:
    #   cont_j = discount * V_{j+1}(S_{j+1})
    # where V_{j+1} is the fitted value function from the next step.
    # Then the optimal value at j for each path is max(K - S_j, cont_j).
    # We fit V_j(S) by regressing these optimal values on S_j.

    # Let's use a simple approach: for each exercise date, create a grid of S values
    # and for each grid point, compute the expected continuation value using the paths.

    # Actually, the cleanest approach for Bermudan with known GBM dynamics:
    # Since we know the model is GBM, we can use the fact that S_{j+1} | S_j = S follows
    # a lognormal distribution. So we can compute the continuation value analytically
    # or via Monte Carlo conditional on S_j.

    # But we have training paths, so let's use them directly.

    # Approach:
    # 1. Start from the last date. V_{n-1}(S) = max(K - S, 0)
    # 2. For j = n-2 down to 0:
    #    a. For each path, compute cont = discount * V_{j+1}(S_{j+1})
    #    b. The optimal value for path i at step j is max(K - S_j[i], cont[i])
    #    c. Fit V_j(S) by regressing these values on S_j
    #    d. Store the exercise threshold: exercise if K - S > V_j(S)

    # For the regression, I'll use a piecewise linear fit on a grid of S values.

    # Define a grid for S values
    S_min = 1.0
    S_max = 300.0
    n_grid = 200
    S_grid = np.linspace(S_min, S_max, n_grid)

    # V[j] will be the value function at exercise date j, evaluated on S_grid
    # V[j] has shape (n_grid,)

    # Last date: V_{n_dates-1}(S) = max(K - S, 0)
    V = np.zeros((n_dates, n_grid))
    V[-1, :] = np.maximum(K - S_grid, 0.0)

    # Backward induction
    for j in range(n_dates - 2, -1, -1):
        if time.time() > deadline:
            # Not enough time, use simple heuristic for remaining dates
            for jj in range(j, -1, -1):
                V[jj, :] = np.maximum(K - S_grid, 0.0)
            break

        # For each path, get S_j and S_{j+1}
        S_j = training_paths[:, j]
        S_j1 = training_paths[:, j + 1]

        # Interpolate V_{j+1} at S_{j+1}
        # Use linear interpolation
        V_j1_at_Sj1 = np.interp(S_j1, S_grid, V[j + 1, :])

        # Continuation value for each path
        cont = discount * V_j1_at_Sj1

        # Exercise payoff for each path
        exercise_payoff = np.maximum(K - S_j, 0.0)

        # Optimal value for each path
        optimal_value = np.maximum(exercise_payoff, cont)

        # Fit V_j(S) by averaging optimal_value in bins of S_j
        # Use binning approach
        bin_edges = np.linspace(S_min, S_max, n_grid + 1)
        bin_indices = np.digitize(S_j, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_grid - 1)

        V_j_fit = np.zeros(n_grid)
        counts = np.zeros(n_grid)
        np.add.at(V_j_fit, bin_indices, optimal_value)
        np.add.at(counts, bin_indices, 1.0)

        # Where we have data, use the average; otherwise, interpolate
        mask = counts > 0
        V_j_fit[mask] /= counts[mask]

        # For bins with no data, interpolate from neighboring bins
        if not mask.all():
            # Create a full interpolation
            interp_indices = np.arange(n_grid)
            valid_indices = np.where(mask)[0]
            if len(valid_indices) >= 2:
                V_j_fit[~mask] = np.interp(interp_indices[~mask], valid_indices, V_j_fit[mask])
            elif len(valid_indices) == 1:
                V_j_fit[~mask] = V_j_fit[valid_indices[0]]
            else:
                V_j_fit[~mask] = np.maximum(K - S_grid[~mask], 0.0)

        V[j, :] = V_j_fit

    # Now we have V[j, :] for each exercise date j.
    # The exercise policy at date j is: exercise if K - S > V[j](S)
    # i.e., exercise if S < threshold_j(S), where threshold is defined implicitly.

    # For the exercise function, we need to return a boolean vector for the current states.
    # The exercise function receives current_state (an array of S values) and time (the current time).
    # We need to map time to the exercise date index.

    # Create the policy object
    class BermudanExercisePolicy:
        def __init__(self, V, S_grid, exercise_times, K, dt):
            self.V = V
            self.S_grid = S_grid
            self.exercise_times = exercise_times
            self.K = K
            self.dt = dt

        def exercise(self, current_state, time):
            # Determine which exercise date we're at
            # time is the current time, exercise dates are at dt, 2dt, ..., n_dates*dt
            # Find the index j such that time is closest to exercise_times[j]
            # Or: j = round(time / dt) - 1, clipped to [0, n_dates-1]
            
            # time is the current time. Exercise dates are at (j+1)*dt for j=0,...,n_dates-1
            # So if time = (j+1)*dt, we're at exercise date j.
            
            # Compute j
            j = int(round(time / self.dt)) - 1
            j = np.clip(j, 0, len(self.exercise_times) - 1)

            # Interpolate V[j] at current_state
            V_at_S = np.interp(current_state, self.S_grid, self.V[j, :])

            # Exercise if K - S > V_at_S
            # i.e., exercise if S < K - V_at_S
            exercise_decision = (self.K - current_state) > V_at_S

            return exercise_decision

    return BermudanExercisePolicy(V, S_grid, exercise_times, K, dt)
# EVOLVE-BLOCK-END
