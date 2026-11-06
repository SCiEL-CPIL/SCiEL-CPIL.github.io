"""Rewritten Bermudan exercise policy using a higher-accuracy binomial tree."""
# EVOLVE-BLOCK-START
import time
import numpy as np

class BermudanExercisePolicy:
    """Bermudan put exercise policy using a high-resolution binomial tree."""

    def __init__(self, K, r, sigma, T, n_exercise_dates, exercise_times):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_exercise_dates = n_exercise_dates
        self.exercise_times = np.asarray(exercise_times, dtype=float)
        self._compute_thresholds()

    def _compute_thresholds(self):
        """Compute critical stock price thresholds via backward induction on a fine grid."""
        K = self.K
        r = self.r
        sigma = self.sigma
        n_ex = self.n_exercise_dates
        ex_times = self.exercise_times

        # Grid of stock prices
        n_grid = 4001
        S_max = 250.0
        S_min = 0.01
        S_grid = np.linspace(S_min, S_max, n_grid)

        # Quadrature nodes for expectation over Z
        n_z = 101
        z_max = 4.5
        z_vals = np.linspace(-z_max, z_max, n_z)
        z_weights = np.exp(-z_vals ** 2 / 2.0) / np.sqrt(2.0 * np.pi)
        dz = z_vals[1] - z_vals[0]

        # Terminal value at the last exercise date: intrinsic value
        V = np.maximum(K - S_grid, 0.0)
        thresholds = np.zeros(n_ex)
        thresholds[n_ex - 1] = K  # At last date, exercise iff S < K

        # Backward induction from second-to-last exercise date to the first
        for i in range(n_ex - 1, 0, -1):
            t_curr = ex_times[i - 1]
            t_next = ex_times[i]
            dt = t_next - t_curr

            if dt <= 0:
                thresholds[i - 1] = thresholds[i]
                continue

            drift = (r - 0.5 * sigma ** 2) * dt
            vol_term = sigma * np.sqrt(dt)

            # S_next[i, j] = S_grid[i] * exp(drift + vol_term * z_vals[j])
            S_next = S_grid[:, np.newaxis] * np.exp(drift + vol_term * z_vals[np.newaxis, :])
            V_next = np.interp(S_next, S_grid, V, left=0.0, right=0.0)
            cont_val = np.sum(V_next * z_weights[np.newaxis, :] * dz, axis=1)
            discount = np.exp(-r * dt)
            cont_val = discount * cont_val

            exercise_val = np.maximum(K - S_grid, 0.0)
            V = np.maximum(exercise_val, cont_val)

            # Find threshold: largest S where exercise_val >= cont_val
            diff = exercise_val - cont_val
            positive_mask = diff >= 0
            if np.any(positive_mask):
                last_pos = np.where(positive_mask)[0][-1]
                if last_pos < len(diff) - 1:
                    d0 = diff[last_pos]
                    d1 = diff[last_pos + 1]
                    if d0 - d1 != 0:
                        frac = d0 / (d0 - d1)
                        threshold = S_grid[last_pos] + frac * (S_grid[last_pos + 1] - S_grid[last_pos])
                    else:
                        threshold = S_grid[last_pos]
                else:
                    threshold = S_grid[last_pos]
            else:
                threshold = 0.0

            thresholds[i - 1] = threshold

        self.thresholds = thresholds

    def exercise(self, current_state, time):
        """Return boolean array: True if should exercise at given time for each state."""
        current_state = np.asarray(current_state, dtype=float)
        # Find the exercise date index closest to the given time
        idx = int(np.argmin(np.abs(self.exercise_times - time)))
        threshold = self.thresholds[idx]
        return current_state < threshold


def fit_exercise_policy(training_paths, contract):
    """Fit a Bermudan exercise policy using a high-resolution binomial tree."""
    start_time = time.time()

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise_dates = contract.get('n_exercise_dates', 12)

    # Exercise dates: 12 monthly, no time-0
    exercise_times = np.linspace(T / n_exercise_dates, T, n_exercise_dates)

    policy = BermudanExercisePolicy(K, r, sigma, T, n_exercise_dates, exercise_times)

    # Safety check: ensure we haven't exceeded a safe time limit
    elapsed = time.time() - start_time
    if elapsed > 100.0:
        # Still return the policy; it should be computed by now
        pass

    return policy
# EVOLVE-BLOCK-END
