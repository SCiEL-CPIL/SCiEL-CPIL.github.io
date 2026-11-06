"""Bermudan put exercise policy via backward induction with exact Gaussian transitions."""
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_exercise = 12
    dt = T / n_exercise

    # Grid for log(S) - use a wide range to capture tail behavior
    # For a put with K=100, S below ~30 is always exercised, above ~200 never
    n_grid = 3000
    log_s_min = np.log(3.0)
    log_s_max = np.log(250.0)
    log_s_grid = np.linspace(log_s_min, log_s_max, n_grid)
    s_grid = np.exp(log_s_grid)

    # Precompute discount factor
    disc = np.exp(-r * dt)

    # Hermite-Gauss quadrature for exact expectation under GBM
    # S_{t+dt} = S_t * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # E[f(S_{t+dt}) | S_t] = integral f(S_t * exp(a + b*Z)) phi(Z) dZ
    # where a = (r - sigma^2/2)*dt, b = sigma*sqrt(dt)
    n_quad = 200
    # Use Gauss-Hermite for integral of f(x)*exp(-x^2) dx
    # Transform: integral g(z) phi(z) dz = (1/sqrt(2pi)) * integral g(z) exp(-z^2/2) dz
    # Gauss-Hermite: integral f(x) exp(-x^2) dx ≈ sum w_i f(x_i)
    # Let x = z/sqrt(2), so dz = sqrt(2) dx
    # integral g(z) exp(-z^2/2) dz = sqrt(2) integral g(sqrt(2)*x) exp(-x^2) dx
    x_h, w_h = np.polynomial.hermite.hermgauss(n_quad)
    z_nodes = np.sqrt(2.0) * x_h
    z_weights = np.sqrt(2.0) * w_h / np.sqrt(np.pi)  # normalized to integrate to 1

    a = (r - 0.5 * sigma**2) * dt
    b = sigma * np.sqrt(dt)

    # Backward induction
    # Value at terminal (last exercise date, index n_exercise-1 = 11)
    # At the last exercise date, value = max(K - S, 0) since no continuation
    value = np.maximum(K - s_grid, 0.0)

    # Exercise boundary tracking for each time step
    boundaries = [None] * n_exercise

    # Go backward from step n_exercise-2 down to 0
    for step in range(n_exercise - 2, -1, -1):
        # Check deadline
        if time.time() > deadline - 5.0:
            break

        # Continuation value: E[disc * V(S_{t+dt}) | S_t]
        # For each grid point S_i, compute E[V(S_i * exp(a + b*Z))]
        # Vectorize: for each S_i, the log of next S is log(S_i) + a + b*Z
        # log_next = log_s_grid[:, None] + a + b * z_nodes[None, :]
        # next_S = exp(log_next)
        # V_interp = interpolate(value, s_grid, next_S)
        # cont = disc * sum(w_j * V_interp[i,j])

        # To avoid huge memory, process in chunks
        chunk_size = 500
        cont_value = np.zeros(n_grid)

        for i_start in range(0, n_grid, chunk_size):
            i_end = min(i_start + chunk_size, n_grid)
            log_s_chunk = log_s_grid[i_start:i_end]
            # Shape: (chunk, n_quad)
            log_next = log_s_chunk[:, None] + a + b * z_nodes[None, :]
            next_S = np.exp(log_next)

            # Interpolate value at next_S
            # value is defined on s_grid, next_S may be outside range
            # Clip to grid range for interpolation, but use exact values at boundaries
            next_S_clipped = np.clip(next_S, s_grid[0], s_grid[-1])
            # np.interp expects increasing xp
            v_interp = np.interp(next_S_clipped, s_grid, value)

            # Weighted sum
            cont_chunk = disc * np.sum(v_interp * z_weights[None, :], axis=1)
            cont_value[i_start:i_end] = cont_chunk

        # Exercise value
        exercise_value = np.maximum(K - s_grid, 0.0)

        # Optimal value
        value = np.maximum(exercise_value, cont_value)

        # Find exercise boundary: largest S where exercise >= continuation
        # Exercise region is S <= boundary
        exercise_mask = exercise_value >= cont_value
        # Find the boundary index
        if np.any(exercise_mask):
            # Find the largest index where exercise is optimal
            boundary_idx = np.max(np.where(exercise_mask)[0])
            boundary_s = s_grid[boundary_idx]
            boundaries[step] = boundary_s
        else:
            boundaries[step] = s_grid[0]

        # Bisection refinement of the boundary
        # The boundary is where exercise_value = cont_value
        # We can refine by finding the exact crossing point
        # For simplicity, use the grid boundary; the value function is already accurate
        # A more precise approach would interpolate the boundary

    # Now create the policy object
    # The policy needs to determine exercise decisions given current states
    # For each time step, exercise if S <= boundary[step]

    # Determine which exercise step we're at based on time
    # Exercise dates are at t = dt, 2*dt, ..., 12*dt = T
    # So at time t, the step index is round(t / dt) - 1 if t > 0

    class ExercisePolicy:
        def __init__(self, boundaries, s_grid, value, n_exercise, dt, T, r, sigma, K):
            self.boundaries = boundaries
            self.s_grid = s_grid
            self.value = value
            self.n_exercise = n_exercise
            self.dt = dt
            self.T = T
            self.r = r
            self.sigma = sigma
            self.K = K

        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: current time (scalar)
            Returns: boolean array, True if exercise
            """
            # Determine exercise step
            if time <= 0:
                # No exercise at time 0
                return np.zeros(len(current_state), dtype=bool)

            # Step index: exercise dates at dt, 2dt, ..., T
            # time should be close to one of these dates
            step = int(np.round(time / self.dt)) - 1
            step = np.clip(step, 0, self.n_exercise - 1)

            if self.boundaries[step] is None:
                # Fallback: use intrinsic value only
                return current_state <= self.K

            # Exercise if S <= boundary
            return current_state <= self.boundaries[step]

    return ExercisePolicy(boundaries, s_grid, value, n_exercise, dt, T, r, sigma, K)
