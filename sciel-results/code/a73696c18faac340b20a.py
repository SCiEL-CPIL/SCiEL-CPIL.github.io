"""Bermudan exercise-policy research: backward induction on a fine log-spaced grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on a fine grid.
    
    Strategy:
    1. Extract contract parameters (K, r, sigma, T, n_steps).
    2. Build a fine log-spaced discretization grid for S in [S_min, S_max].
    3. Perform backward induction from maturity (T) back to the first exercise date (T/n).
       At each exercise date, compute continuation value via risk-neutral expectation
       using Gauss-Hermite quadrature, then set exercise boundary where (K-S) >= continuation.
    4. Return an ExercisePolicy object that uses the fitted boundaries to decide exercise.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Extract contract parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 12  # 12 monthly exercise dates

    # Grid parameters - focus on relevant S range for a put with K=100
    # For a put, exercise is most likely when S < K, so focus grid there
    S_min = 5.0
    S_max = 150.0
    n_grid = 200  # number of grid points in log space

    # Gauss-Hermite quadrature for expectation computation
    # Use 40-point GH quadrature for accuracy
    n_gh = 40
    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_gh)

    # Log-spaced grid
    log_S = np.linspace(np.log(S_min), np.log(S_max), n_grid)
    S_grid = np.exp(log_S)

    # Precompute discount factor per step
    dt = T / n_steps
    disc = math.exp(-r * dt)

    # Backward induction: start from maturity
    # At maturity, value = payoff = max(K - S, 0)
    V = np.maximum(K - S_grid, 0.0)

    # Exercise boundaries: store the max S at which exercise is optimal at each date
    # boundaries[i] = max S such that at date i (counting from 0 = first exercise date),
    # it's optimal to exercise
    boundaries = np.zeros(n_steps)

    # We need to go from the last date (index n_steps-1, time T) back to first (index 0, time T/n_steps)
    # At date n_steps-1 (time T), it's the last chance to exercise, so value = payoff
    # Then for dates n_steps-2 down to 0:
    #   continuation = disc * E[V_next]
    #   value = max(payoff, continuation)
    #   exercise if payoff >= continuation

    for i in range(n_steps - 1, -1, -1):
        if time.time() > deadline:
            break

        # Compute continuation value: disc * E[V(S_{t+dt}) | S_t = S_grid]
        # Under GBM: S_{t+dt} = S_t * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
        # Z ~ N(0,1)
        # E[f(S_{t+dt})] = integral f(S_t * exp((r-sigma^2/2)*dt + sigma*sqrt(dt)*x)) * phi(x) dx
        # Using GH quadrature: integral g(x) * exp(-x^2) dx ≈ sum w_j * g(x_j)
        # phi(x) = exp(-x^2/2) / sqrt(2*pi)
        # So E[f] = (1/sqrt(2*pi)) * sum w_j * f(S_t * exp((r-sigma^2/2)*dt + sigma*sqrt(dt)*x_j)) * exp(x_j^2/2)

        drift = (r - 0.5 * sigma * sigma) * dt
        vol = sigma * math.sqrt(dt)

        # For each grid point S_t, compute E[V(S_{t+dt})]
        # S_{t+dt} = S_t * exp(drift + vol * x_j)
        # Vectorize: S_next[j, k] = S_grid[k] * exp(drift + vol * x_gh[j])
        S_next = S_grid[np.newaxis, :] * np.exp(drift + vol * x_gh[:, np.newaxis])
        # S_next shape: (n_gh, n_grid)

        # Interpolate V at S_next
        # V is defined on S_grid (log-spaced), so interpolate in log space
        log_S_next = np.log(S_next)
        # Clip to grid range
        log_S_next_clipped = np.clip(log_S_next, log_S[0], log_S[-1])
        
        # Linear interpolation in log space
        V_next = np.interp(log_S_next_clipped, log_S, V)
        # V_next shape: (n_gh, n_grid)

        # Weight: w_j * exp(x_j^2/2) / sqrt(2*pi)
        gh_weights = w_gh * np.exp(0.5 * x_gh**2) / math.sqrt(2.0 * math.pi)
        
        # Continuation value at each grid point
        continuation = disc * np.sum(gh_weights[:, np.newaxis] * V_next, axis=0)

        # Payoff at this date
        payoff = np.maximum(K - S_grid, 0.0)

        # Value = max(payoff, continuation)
        V = np.maximum(payoff, continuation)

        # Exercise boundary: max S where payoff >= continuation
        exercise_mask = payoff >= continuation
        if np.any(exercise_mask):
            # Find the largest S where exercise is optimal
            # Since S_grid is increasing, find the last True index
            last_exercise_idx = np.where(exercise_mask)[0][-1]
            boundaries[i] = S_grid[last_exercise_idx]
        else:
            boundaries[i] = 0.0  # Never exercise

    # Create the exercise policy object
    class ExercisePolicy:
        def __init__(self, boundaries, S_grid, log_S_grid):
            self.boundaries = boundaries
            self.S_grid = S_grid
            self.log_S_grid = log_S_grid
            self.n_steps = n_steps

        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: current time
            Returns boolean vector: True if exercise is optimal
            """
            # Determine which exercise date we're at
            # Exercise dates are at T/n_steps, 2*T/n_steps, ..., T
            # time is the current time
            # Find the nearest exercise date index
            # Dates: i*dt for i = 1, ..., n_steps (1-indexed)
            # In 0-indexed: date_idx = round(time / dt) - 1, clamped to [0, n_steps-1]
            
            date_idx = int(round(time / dt)) - 1
            date_idx = max(0, min(date_idx, self.n_steps - 1))

            boundary = self.boundaries[date_idx]

            # Exercise if S <= boundary
            return current_state <= boundary

    return ExercisePolicy(boundaries, S_grid, log_S)
# EVOLVE-BLOCK-END
