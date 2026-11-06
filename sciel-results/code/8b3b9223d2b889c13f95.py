"""Bermudan exercise-policy: Longstaff-Schwartz regression-based backward induction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from numpy.polynomial import polynomial as P

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Parse contract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    n_train = training_paths.shape[0]
    n_steps = n_dates  # monthly steps

    # Exercise times (no time-0 exercise, so dates are at T/12, 2T/12, ..., 12T/12)
    dt = T / n_steps
    exercise_times = np.linspace(dt, T, n_steps)

    # Discount factors
    discount = np.exp(-r * exercise_times)

    # Generate training paths if not provided, or use provided
    # training_paths shape: (n_train, n_steps) - stock prices at each exercise date
    if training_paths is None or len(training_paths) == 0:
        # Generate antithetic pairs
        rng = np.random.default_rng(42)
        half = n_train // 2
        z = rng.standard_normal((half, n_steps))
        z_anti = -z
        z_all = np.vstack([z, z_anti])
        log_drift = (r - 0.5 * sigma**2) * dt
        vol_step = sigma * np.sqrt(dt)
        log_S = np.log(S0) + np.cumsum(log_drift + vol_step * z_all, axis=1)
        training_paths = np.exp(log_S)
        n_train = training_paths.shape[0]

    # Ensure we have enough paths
    n_train = training_paths.shape[0]

    # Fit polynomial regression coefficients for continuation value at each step
    # Using backward induction (Longstaff-Schwartz style)
    # V[i] = max(K - S_i, E[discount * V[i+1] | S_i])
    # At maturity: V[n-1] = max(K - S_{n-1}, 0)

    # We'll fit polynomials to estimate E[V[i+1] | S_i]
    # Use degree-4 polynomial regression in log(S)

    max_degree = 4
    n_poly = max_degree + 1

    # Initialize continuation values at final step
    # V_final = max(K - S_final, 0)
    S_final = training_paths[:, -1]
    V_next = np.maximum(K - S_final, 0.0)

    # Store fitted polynomials for each step
    # poly_coeffs[i] contains coefficients for polynomial in x = log(S)
    # V_cont(x) = sum_j coeffs[j] * x^j
    poly_coeffs = [None] * n_steps

    # Backward induction
    for i in range(n_steps - 2, -1, -1):
        # Check deadline
        if time.time() > deadline - 5.0:
            # Not enough time, use simple threshold for remaining steps
            break

        S_i = training_paths[:, i]
        x = np.log(S_i)

        # Fit polynomial to V_next as function of x
        # V_next is the continuation value from next step (already includes discount)
        # We need E[discount * V[i+1] | S_i], but V_next already has the payoff at i+1
        # Actually, in backward induction:
        # V[i] = max(K - S_i, exp(-r*dt) * E[V[i+1] | S_i])
        # So we fit E[V[i+1] | S_i] first, then multiply by discount

        # Fit polynomial: V_next ≈ sum_j c_j * x^j
        # Use least squares
        X = np.vander(x, N=n_poly, increasing=True)  # x^0, x^1, ..., x^max_degree
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, V_next, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(n_poly)

        poly_coeffs[i] = coeffs

    # Now create the exercise policy
    # At each step i, exercise if (K - S_i) >= exp(-r*dt) * fitted_continuation(S_i)
    # i.e., exercise if (K - S_i) >= discount[i] * poly_eval(log(S_i))

    class BermudanExercisePolicy:
        def __init__(self, poly_coeffs, discount, dt, K, n_steps):
            self.poly_coeffs = poly_coeffs
            self.discount = discount
            self.dt = dt
            self.K = K
            self.n_steps = n_steps
            self.current_step = 0

        def exercise(self, current_state, time):
            """
            current_state: array of stock prices
            time: current time (to determine which exercise date)
            Returns: boolean array, True if exercise
            """
            # Determine which exercise date we're at
            # Exercise dates are at dt, 2dt, ..., n_steps*dt
            # time should match one of these
            if time is None or time <= 0:
                return np.zeros(len(current_state), dtype=bool)

            # Find the step index
            step_idx = int(round(time / self.dt)) - 1
            step_idx = max(0, min(step_idx, self.n_steps - 1))

            # Get the polynomial coefficients for this step
            if step_idx < len(self.poly_coeffs) and self.poly_coeffs[step_idx] is not None:
                coeffs = self.poly_coeffs[step_idx]
            else:
                # Fallback: use simple threshold
                # For a put, exercise when S is low enough
                # Approximate threshold: exercise when S < K * 0.85
                return current_state < self.K * 0.85

            # Evaluate continuation value
            x = np.log(current_state)
            # Evaluate polynomial: sum_j coeffs[j] * x^j
            cont_value = np.zeros_like(x)
            for j in range(len(coeffs)):
                cont_value += coeffs[j] * (x ** j)

            # Discounted continuation value
            disc_cont = self.discount[step_idx] * cont_value

            # Immediate exercise payoff
            exercise_payoff = np.maximum(self.K - current_state, 0.0)

            # Exercise if immediate payoff >= discounted continuation
            return exercise_payoff >= disc_cont

    # Build the policy
    policy = BermudanExercisePolicy(poly_coeffs, discount, dt, K, n_steps)
    return policy

# EVOLVE-BLOCK-END
