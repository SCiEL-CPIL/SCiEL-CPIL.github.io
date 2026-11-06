"""Bermudan exercise policy using Monte Carlo Longstaff-Schwartz regression approach."""
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12

    # Exercise dates: t_k = k * T / n_exercise for k=1,...,12
    dt = T / n_exercise
    exercise_times = np.arange(1, n_exercise + 1) * dt

    # Get training paths
    if training_paths is None or len(training_paths) == 0:
        # Fallback to tree-based approach if no training data
        return _tree_policy(K, r, sigma, T, n_exercise)

    n_paths, n_time_steps = training_paths.shape

    # Determine time grid from training paths
    if n_time_steps > 1:
        dt_train = T / (n_time_steps - 1)
    else:
        dt_train = T

    # Map exercise dates to training path indices
    ex_indices = []
    for k in range(1, n_exercise + 1):
        t_ex = k * dt
        idx = int(round(t_ex / dt_train))
        idx = max(0, min(idx, n_time_steps - 1))
        ex_indices.append(idx)

    # Longstaff-Schwartz backward induction
    # At each exercise date from last to first:
    # 1. At maturity: continuation = 0, so exercise if payoff > 0
    # 2. At earlier dates: fit regression of continuation value on current state
    #    using paths that did NOT exercise at later dates

    # We'll track which paths are still "alive" (haven't exercised yet)
    # and their continuation values

    # Start at maturity (exercise date n_exercise)
    # Continuation value at maturity is 0
    # Exercise if payoff > 0

    # Store the exercise decision function for each date
    # We'll use polynomial regression: continuation ≈ f(S)
    # For a put, we expect continuation to be decreasing in S

    # At each step, we need:
    # - States of alive paths at this exercise date
    # - Continuation values (discounted expected value of future optimal exercise)
    # - Fit regression: cont_val ≈ poly(state)
    # - Exercise where payoff > fitted continuation

    # Work backward
    # alive_mask: which paths haven't exercised yet
    alive_mask = np.ones(n_paths, dtype=bool)

    # continuation_values: for alive paths, the estimated continuation value
    # At maturity, continuation = 0 for all
    continuation_values = np.zeros(n_paths)

    # Store fitted polynomials for each exercise date
    # poly_coeffs[k] = coefficients for polynomial at exercise date k
    # We'll use degree 4 polynomial: a0 + a1*S + a2*S^2 + a3*S^3 + a4*S^4
    POLY_DEG = 4
    poly_coeffs = {}  # key: exercise date index (1-12), value: array of coefficients

    # Work backward from last exercise date to first
    for k in range(n_exercise, 0, -1):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            # Return whatever we have so far
            break

        ex_idx = ex_indices[k - 1]
        states = training_paths[:, ex_idx]

        if k == n_exercise:
            # At maturity: continuation = 0, exercise if payoff > 0
            payoff = np.maximum(K - states, 0.0)
            exercise_mask = payoff > 0

            # Update alive mask
            alive_mask = alive_mask & ~exercise_mask

            # Continuation for alive paths is 0 (maturity)
            continuation_values = np.zeros(n_paths)

            # Fit a trivial polynomial (all zeros) or just store the decision rule
            # For the policy, at maturity we just exercise if S < K
            poly_coeffs[k] = np.zeros(POLY_DEG + 1)  # zero polynomial
        else:
            # For alive paths, we have continuation values from the next step
            # Discount them back
            discount = np.exp(-r * dt)
            cont_vals_alive = discount * continuation_values[alive_mask]
            states_alive = states[alive_mask]

            if np.sum(alive_mask) < POLY_DEG + 2:
                # Not enough data for regression, fall back to simple rule
                payoff = np.maximum(K - states, 0.0)
                exercise_mask = payoff > 0
                alive_mask = alive_mask & ~exercise_mask
                poly_coeffs[k] = np.zeros(POLY_DEG + 1)
                continue

            # Fit polynomial regression: cont_val ≈ f(S)
            # Use numpy polyfit
            try:
                # For numerical stability, scale S
                S_scale = 100.0  # K
                S_scaled = states_alive / S_scale
                coeffs = np.polyfit(S_scaled, cont_vals_alive, POLY_DEG)
                poly_coeffs[k] = coeffs
            except Exception:
                # Fallback: linear fit
                try:
                    coeffs = np.polyfit(states_alive / S_scale, cont_vals_alive, 1)
                    # Pad to POLY_DEG
                    padded = np.zeros(POLY_DEG + 1)
                    padded[:len(coeffs)] = coeffs
                    poly_coeffs[k] = padded
                except Exception:
                    poly_coeffs[k] = np.zeros(POLY_DEG + 1)

            # Evaluate fitted continuation for ALL paths at this date
            S_all_scaled = states / S_scale
            cont_fitted = np.polyval(poly_coeffs[k], S_all_scaled)
            # Ensure continuation is non-negative
            cont_fitted = np.maximum(cont_fitted, 0.0)

            # Exercise where payoff > fitted continuation
            payoff = np.maximum(K - states, 0.0)
            exercise_mask = (payoff > cont_fitted) & alive_mask

            # Update alive mask
            alive_mask = alive_mask & ~exercise_mask

            # Update continuation values for alive paths
            # The continuation value for alive paths is the fitted value
            # (which already includes the discount from the next step)
            continuation_values[alive_mask] = cont_fitted[alive_mask]

    # Build the policy object
    class ExercisePolicy:
        def __init__(self):
            self.poly_coeffs = poly_coeffs
            self.n_exercise = n_exercise
            self.T = T
            self.K = K
            self.S_scale = 100.0

        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: current time
            Returns: boolean array, True if exercise
            """
            if time <= 0 or time > self.T:
                return np.zeros(len(current_state), dtype=bool)

            # Find the exercise date index
            k_float = time * self.n_exercise / self.T
            k = int(round(k_float))
            k = max(1, min(k, self.n_exercise))

            # Check if close to an exercise date
            t_ex = k * self.T / self.n_exercise
            if abs(time - t_ex) > 0.01 * self.T:
                return np.zeros(len(current_state), dtype=bool)

            payoff = np.maximum(self.K - current_state, 0.0)

            if k == self.n_exercise:
                # At maturity: exercise if payoff > 0
                return payoff > 0

            # Evaluate fitted continuation
            coeffs = self.poly_coeffs.get(k, np.zeros(POLY_DEG + 1))
            S_scaled = current_state / self.S_scale
            cont_fitted = np.polyval(coeffs, S_scaled)
            cont_fitted = np.maximum(cont_fitted, 0.0)

            return payoff > cont_fitted

    return ExercisePolicy()


def _tree_policy(K, r, sigma, T, n_exercise):
    """Fallback binomial tree policy if no training data available."""
    dt_sparse = T / n_exercise
    u = np.exp(sigma * np.sqrt(dt_sparse))
    d = 1.0 / u
    p = (np.exp(r * dt_sparse) - d) / (u - d)
    discount = np.exp(-r * dt_sparse)

    S0 = 100.0

    # Backward induction
    n_final = n_exercise
    k_mat = np.arange(n_final + 1)
    S_mat = S0 * u ** (2 * k_mat - n_final)
    current_values = np.maximum(K - S_mat, 0.0)

    critical_prices = {}

    for n in range(n_final - 1, 0, -1):
        up_val = current_values[1:n + 2]
        down_val = current_values[0:n + 1]
        cont_val = discount * (p * up_val + (1 - p) * down_val)

        k_nodes = np.arange(n + 1)
        S_nodes = S0 * u ** (2 * k_nodes - n)
        ex_val = np.maximum(K - S_nodes, 0.0)

        exercise_mask = ex_val > cont_val
        exercising_indices = np.where(exercise_mask)[0]

        if len(exercising_indices) > 0:
            max_k = exercising_indices[-1]
            if max_k < n:
                S_ex = S_nodes[max_k]
                S_no_ex = S_nodes[max_k + 1]
                diff_k = ex_val[max_k] - cont_val[max_k]
                diff_k1 = ex_val[max_k + 1] - cont_val[max_k + 1]
                if diff_k > 0 and diff_k1 < 0:
                    frac = diff_k / (diff_k - diff_k1)
                    cp = S_ex + frac * (S_no_ex - S_ex)
                elif diff_k > 0:
                    cp = S_no_ex
                else:
                    cp = S_ex
            else:
                cp = S_nodes[max_k]
            critical_prices[n] = cp
        else:
            critical_prices[n] = 0.0

        current_values = np.maximum(ex_val, cont_val)

    class ExercisePolicy:
        def __init__(self):
            self.critical_prices = critical_prices
            self.n_exercise = n_exercise
            self.T = T
            self.K = K

        def exercise(self, current_state, time):
            if time <= 0 or time > self.T:
                return np.zeros(len(current_state), dtype=bool)

            k_float = time * self.n_exercise / self.T
            k = int(round(k_float))
            k = max(1, min(k, self.n_exercise))

            t_ex = k * self.T / self.n_exercise
            if abs(time - t_ex) > 0.01 * self.T:
                return np.zeros(len(current_state), dtype=bool)

            cp = self.critical_prices.get(k, 0.0)
            return current_state < cp

    return ExercisePolicy()
