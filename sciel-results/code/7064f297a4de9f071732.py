"""Bermudan put exercise policy: backward induction with refined discretization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction.
    
    Parameters:
    -----------
    training_paths : array-like, shape (n_paths, n_time_steps)
        Simulated spot price paths. n_time_steps = 13 (t=0..12).
    contract : dict
        Contract parameters: K, r, sigma, T, n_steps (12 exercise dates).
        
    Returns:
    --------
    An object with method exercise(current_state, time) -> boolean array.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Extract contract parameters
    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))
    n_steps = int(contract.get('n_steps', 12))

    # Exercise dates: 12 monthly dates, no time-0 exercise
    # t_k = k * T / n_steps for k = 1, ..., n_steps
    dt = T / n_steps
    exercise_times = np.array([k * dt for k in range(1, n_steps + 1)])

    # Discount factors
    disc = np.exp(-r * dt)

    # We'll use a grid-based approach with backward induction
    # Determine grid from training paths
    tp = np.asarray(training_paths, dtype=np.float64)
    if tp.ndim == 1:
        tp = tp.reshape(1, -1)
    n_paths, n_ts = tp.shape

    # The paths have n_ts time points (should be 13: t=0..12)
    # Exercise decisions at indices 1..12 (12 exercise dates)

    # Strategy: Use a fine grid of spot prices and compute continuation
    # values via backward induction. Then the policy is: exercise if
    # intrinsic value >= continuation value.

    # Build grid from training data range, but extend for safety
    all_spots = tp[:, 1:]  # spots at exercise times
    min_s = max(1.0, np.min(all_spots) * 0.5)
    max_s = max(K * 3.0, np.max(all_spots) * 1.5)

    # Use a logarithmic grid for better resolution at low prices (important for puts)
    n_grid = 2048
    log_min = np.log(min_s)
    log_max = np.log(max_s)
    log_grid = np.linspace(log_min, log_max, n_grid)
    grid = np.exp(log_grid)  # spot price grid

    # Backward induction: compute continuation value at each exercise date
    # V_k(s) = max(K - s, disc * E[V_{k+1}(S_{k+1}) | S_k = s])
    # For the last exercise date, V_{n_steps}(s) = max(K - s, 0)

    # We'll represent the value function on the grid at each step
    # V is an array of shape (n_grid,) for each step

    # Payoff at exercise: max(K - s, 0)
    def payoff(s):
        return np.maximum(K - s, 0.0)

    # For risk-neutral GBM: S_{k+1} = S_k * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    # We need E[V_{k+1}(S_{k+1}) | S_k = s] for each s on our grid.
    # This is a convolution-like operation. We can compute it via:
    # S_{k+1} = s * exp(mu*dt + sigma*sqrt(dt)*Z) where mu = r - 0.5*sigma^2
    # So ln(S_{k+1}) = ln(s) + mu*dt + sigma*sqrt(dt)*Z
    # Z ~ N(0,1)

    mu = r - 0.5 * sigma * sigma
    vol_step = sigma * np.sqrt(dt)

    # Precompute the transition: for each grid point s_i, S_{k+1} is lognormal
    # centered at ln(s_i) + mu*dt with std vol_step.
    # E[V_{k+1}(S_{k+1})] = integral V_{k+1}(s * exp(mu*dt + vol_step*z)) phi(z) dz
    # We can approximate this using a Gauss-Hermite quadrature or simple Monte Carlo.
    # For speed, use a fixed set of quadrature points.

    # Use Gauss-Legendre quadrature with 64 points on [-8, 8]
    n_quad = 64
    # Transform to standard normal: use Gauss-Hermite
    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_quad)
    # Hermite quadrature: integral f(x) exp(-x^2) dx = sum w_i f(x_i)
    # We want integral f(z) phi(z) dz = (1/sqrt(2pi)) integral f(z) exp(-z^2/2) dz
    # Let z = sqrt(2)*x, then dz = sqrt(2) dx, exp(-z^2/2) = exp(-x^2)
    # So integral f(z) phi(z) dz = (1/sqrt(2pi)) * sqrt(2) * integral f(sqrt(2)*x) exp(-x^2) dx
    # = (1/sqrt(pi)) * integral f(sqrt(2)*x) exp(-x^2) dx
    # = (1/sqrt(pi)) * sum w_i f(sqrt(2)*x_i)

    z_quad = np.sqrt(2.0) * x_gh
    w_quad = w_gh / np.sqrt(np.pi)  # normalized so sum w_quad * f(z_quad) approx E[f(Z)]

    # Verify: sum(w_quad) should be ~1
    # sum(w_gh) = sqrt(pi), so sum(w_quad) = 1. Good.

    # Backward induction
    # V_next: value function at the next step, on the grid
    # At the last exercise date (k = n_steps), the value is just the payoff
    # (no further continuation)

    # V at step k represents the value BEFORE exercising at step k
    # V_k(s) = max(payoff(s), disc * E[V_{k+1}(S_{k+1}) | S_k = s])
    # At k = n_steps (last exercise), V_{n_steps}(s) = payoff(s)
    # At k < n_steps, V_k(s) = max(payoff(s), disc * E[V_{k+1}(S_{k+1}) | S_k = s])

    # We store the continuation value for each step to build the policy
    # Policy at step k: exercise if payoff(s) >= disc * E[V_{k+1}(S_{k+1}) | S_k = s]

    # Let's compute from the last step backwards
    # V_cont[k] = continuation value at step k = disc * E[V_{k+1}(S_{k+1}) | S_k]
    # for k = 1, ..., n_steps-1
    # V_cont[n_steps] is not needed (no continuation after last exercise)

    # Actually, let's think about this more carefully.
    # Exercise dates: k = 1, 2, ..., n_steps
    # At date k, we decide whether to exercise.
    # If we exercise at date k, we get payoff(S_k) = max(K - S_k, 0)
    # If we don't exercise, we get the continuation value = E[disc * V_{k+1}(S_{k+1}) | S_k]
    # where V_{k+1} is the value at the next exercise date.

    # V_k(s) = max(payoff(s), disc * E[V_{k+1}(S_{k+1}) | S_k = s])
    # V_{n_steps}(s) = payoff(s)  (must exercise or let it expire; no further dates)

    # So we compute V from k = n_steps down to k = 1.
    # The policy at date k is: exercise if payoff(s) >= disc * E[V_{k+1}(S_{k+1}) | S_k = s]

    # Store the continuation values for the policy
    # cont_vals[k] = disc * E[V_{k+1}(S_{k+1}) | S_k = s] for each s on grid
    # for k = 1, ..., n_steps-1
    # For k = n_steps, there's no continuation, so always exercise if payoff > 0

    n_ex = n_steps  # number of exercise dates
    # cont_vals is a list of length n_ex, where cont_vals[k-1] is the continuation
    # value at exercise date k (0-indexed: date 1 -> index 0, ..., date n_steps -> index n_steps-1)
    # For the last date, continuation is 0 (no more dates)

    # V_next: value function at the next exercise date
    # Start with V at the last exercise date
    V_next = payoff(grid)  # V_{n_steps}(s) = payoff(s)

    # Store continuation values for each exercise date
    # cont_at_date[k] for k = 0, ..., n_ex-1 (corresponding to dates 1, ..., n_ex)
    cont_at_date = [None] * n_ex

    # Process dates from last to first
    for k_idx in range(n_ex - 1, -1, -1):
        # k_idx = 0 -> date 1, k_idx = n_ex-1 -> date n_ex
        # At this date, V_next is the value at the NEXT date
        # If this is the last date (k_idx = n_ex-1), there is no next date
        if k_idx == n_ex - 1:
            # Last exercise date: no continuation
            cont_at_date[k_idx] = np.zeros(n_grid)
            V_current = V_next  # V_current = payoff, which is V_next
        else:
            # Compute continuation: disc * E[V_next(S_{k+1}) | S_k = s]
            # For each s_i on grid, compute E[V_next(s_i * exp(mu*dt + vol_step*Z))]
            # = sum_j w_quad * V_next(s_i * exp(mu*dt + vol_step*z_j))

            # log_S_next = log(s_i) + mu*dt + vol_step*z_j
            # log_s_i is log_grid
            # log_S_next[i,j] = log_grid[i] + mu*dt + vol_step*z_quad[j]
            # S_next[i,j] = exp(log_grid[i] + mu*dt + vol_step*z_quad[j])

            # For efficiency, compute for all grid points at once
            # log_S_next has shape (n_grid, n_quad)
            log_S_next = log_grid[:, np.newaxis] + mu * dt + vol_step * z_quad[np.newaxis, :]
            S_next = np.exp(log_S_next)

            # Now we need to evaluate V_next at S_next
            # V_next is defined on the grid (log_grid). We need to interpolate.
            # Use linear interpolation in log-space.
            # V_next_interp = np.interp(log_S_next, log_grid, V_next)
            # But np.interp works on 1D. We can flatten.
            # Alternatively, use a vectorized approach.

            # Flatten for interpolation
            log_S_next_flat = log_S_next.ravel()
            V_next_flat = np.interp(log_S_next_flat, log_grid, V_next)
            V_next_interp = V_next_flat.reshape(n_grid, n_quad)

            # Continuation value at each grid point
            cont_vals = disc * np.sum(w_quad[np.newaxis, :] * V_next_interp, axis=1)
            cont_at_date[k_idx] = cont_vals

            # V_current = max(payoff, continuation)
            V_current = np.maximum(payoff(grid), cont_vals)

        # V_current becomes V_next for the next iteration (going backwards)
        V_next = V_current

        # Check time
        if time.time() > deadline - 5.0:
            break

    # Now we have cont_at_date[k_idx] for each exercise date
    # The policy: exercise at date k (1-indexed) if payoff(s) >= cont_at_date[k-1](s)
    # Note: cont_at_date[k-1] is the continuation value, and payoff is the exercise value.
    # Exercise if payoff >= continuation.

    # Build the exercise policy object
    class ExercisePolicy:
        def __init__(self, grid, log_grid, cont_at_date, payoff_func, K_val):
            self.grid = grid
            self.log_grid = log_grid
            self.cont_at_date = cont_at_date
            self.payoff_func = payoff_func
            self.K = K_val
            self.n_grid = n_grid

        def exercise(self, current_state, time_idx):
            """
            Determine whether to exercise for each path at the given time.
            
            Parameters:
            -----------
            current_state : array-like, shape (n_paths,)
                Current spot prices.
            time_idx : int
                Exercise date index (1 to n_steps). 1-based.
                
            Returns:
            --------
            Boolean array of shape (n_paths,).
            """
            s = np.asarray(current_state, dtype=np.float64)
            
            # time_idx is 1-based: 1 to n_steps
            # Map to 0-based index in cont_at_date
            k_idx = int(time_idx) - 1
            if k_idx < 0 or k_idx >= n_ex:
                # Out of range: default to not exercise
                return np.zeros(len(s), dtype=bool)

            # Get continuation values on the grid
            cont_grid = self.cont_at_date[k_idx]
            if cont_grid is None:
                return np.zeros(len(s), dtype=bool)

            # Interpolate continuation values at current states
            # Use log-space interpolation
            log_s = np.log(np.maximum(s, 1e-10))
            cont_vals = np.interp(log_s, self.log_grid, cont_grid)

            # Exercise if payoff >= continuation
            payoffs = np.maximum(self.K - s, 0.0)
            return payoffs >= cont_vals

    return ExercisePolicy(grid, log_grid, cont_at_date, payoff, K)


# EVOLVE-BLOCK-END
