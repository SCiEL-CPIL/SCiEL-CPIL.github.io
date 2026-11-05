"""Bermudan put exercise-policy research with backward induction on training paths."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

class ExercisePolicy:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def exercise(self, current_state, time_index):
        """
        current_state: array of stock prices
        time_index: integer index of the exercise date (0 to 11)
        Returns: boolean array
        """
        if time_index < 0 or time_index >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
        return current_state < self.thresholds[time_index]

def fit_exercise_policy(training_paths, contract):
    start_time = _time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Determine shape of training_paths
    # Task says 4096 training paths and 12 monthly exercise dates
    # Could be (n_paths, n_dates) or (n_dates, n_paths)
    if training_paths.ndim == 2:
        if training_paths.shape[0] == 4096:
            # (n_paths, n_dates)
            n_paths, n_dates = training_paths.shape
        elif training_paths.shape[1] == 4096:
            # (n_dates, n_paths)
            n_dates, n_paths = training_paths.shape
            training_paths = training_paths.T  # Convert to (n_paths, n_dates)
        else:
            # Fallback: assume (n_paths, n_dates)
            n_paths, n_dates = training_paths.shape
    else:
        raise ValueError("training_paths must be 2D")

    # Get parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    T = contract.get('T', 1.0)
    dt = T / n_dates
    discount_factor = np.exp(-r * dt)

    # training_paths is now (n_paths, n_dates)
    # n_dates should be 12

    # Initialize values
    V = np.zeros((n_paths, n_dates))

    # Terminal payoff at last exercise date
    V[:, -1] = np.maximum(K - training_paths[:, -1], 0.0)

    # Backward Induction
    n_bins = 100  # Number of bins for continuation value estimation

    for t in range(n_dates - 2, -1, -1):
        S_t = training_paths[:, t]
        V_next = V[:, t + 1]

        # Estimate E[V_next | S_t] using binning
        # Sort indices by S_t
        sorted_indices = np.argsort(S_t)
        sorted_S = S_t[sorted_indices]
        sorted_V_next = V_next[sorted_indices]

        # Create bin edges using quantiles for roughly equal bin sizes
        bin_edges = np.quantile(sorted_S, np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(bin_edges)

        # Assign each sorted point to a bin
        bin_indices = np.digitize(sorted_S, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)

        # Calculate mean V_next for each bin
        bin_sums = np.zeros(len(bin_edges) - 1)
        bin_counts = np.zeros(len(bin_edges) - 1)

        np.add.at(bin_sums, bin_indices, sorted_V_next)
        np.add.at(bin_counts, bin_indices, 1)

        # Avoid division by zero
        bin_counts[bin_counts == 0] = 1
        bin_means = bin_sums / bin_counts

        # For each path, get the bin mean corresponding to its S_t
        cont_values_sorted = bin_means[bin_indices]

        # Map back to original order
        cont_values = np.zeros(n_paths)
        cont_values[sorted_indices] = cont_values_sorted

        # Discount the continuation value
        cont_values *= discount_factor

        # Payoff if exercised now
        payoff_now = np.maximum(K - S_t, 0.0)

        # Optimal value
        V[:, t] = np.maximum(payoff_now, cont_values)

        # Check deadline
        if _time.time() > deadline - 5.0:
            break

    # Determine thresholds
    thresholds = np.zeros(n_dates)

    for t in range(n_dates):
        S_t = training_paths[:, t]
        payoff_now = np.maximum(K - S_t, 0.0)

        # Find where exercise is optimal: V[:, t] == payoff_now (within tolerance)
        tol = 1e-8
        is_exercise = np.abs(V[:, t] - payoff_now) < tol
        is_exercise = is_exercise & (payoff_now > 0)

        if np.any(is_exercise):
            thresholds[t] = np.max(S_t[is_exercise])
        else:
            thresholds[t] = 0.0

    return ExercisePolicy(thresholds)
# EVOLVE-BLOCK-END
