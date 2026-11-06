"""Combined Bermudan put exercise policy: refined backward induction with adaptive binning and smoothing."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on an adaptive grid.
    Uses refined quantile binning, smoothing, and careful continuation value estimation.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    # Extract parameters from contract
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', contract.get('maturity', 1.0))
    n_dates = contract.get('n_dates', contract.get('n_exercise', 12))

    # Training paths shape: (n_paths, n_dates)
    if training_paths is None:
        n_paths = 4096
        training_paths = np.zeros((n_paths, n_dates))
    else:
        training_paths = np.asarray(training_paths, dtype=np.float64)
        if training_paths.ndim == 1:
            training_paths = training_paths.reshape(-1, n_dates)
        n_paths, actual_n_dates = training_paths.shape
        n_dates = actual_n_dates

    dt = T / n_dates
    disc = math.exp(-r * dt)

    # Use quantile-based binning for better coverage
    # Start with a reasonable number of bins, then refine
    n_bins = 96  # Slightly more bins for better resolution

    # Compute bin edges from training data using quantiles
    all_prices = training_paths.flatten()
    positive_prices = all_prices[all_prices > 0]
    if len(positive_prices) < n_bins:
        price_min = max(0.01, all_prices.min() * 0.9 if all_prices.size > 0 else 0.01)
        price_max = all_prices.max() * 1.1 if all_prices.size > 0 else 200.0
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    else:
        # Use quantiles for better spacing
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(positive_prices, quantiles)
        # Ensure strictly increasing
        for i in range(1, len(bin_edges)):
            if bin_edges[i] <= bin_edges[i-1]:
                bin_edges[i] = bin_edges[i-1] + 1e-6
        # Extend edges slightly
        bin_edges[0] = max(0.01, bin_edges[0] * 0.95)
        bin_edges[-1] = bin_edges[-1] * 1.05

    # Backward induction
    # Start at last exercise date: V = max(K - S, 0)
    V_current = np.maximum(K - training_paths[:, -1], 0.0)

    # Store continuation value estimates as bin averages for each date
    cont_per_date = [None] * (n_dates - 1)

    # Process dates from n_dates-2 down to 0
    for i in range(n_dates - 2, -1, -1):
        if time.time() > deadline - 5:
            # Break early, use default policy for remaining dates
            break

        S_i = training_paths[:, i]

        # Bin the S_i values
        bin_indices = np.digitize(S_i, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Compute mean V_current (which is V_{i+1}) for each bin
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        np.add.at(bin_sums, bin_indices, V_current)
        np.add.at(bin_counts, bin_indices, 1)

        bin_counts_safe = np.where(bin_counts > 0, bin_counts, 1)
        bin_means = bin_sums / bin_counts_safe

        # For bins with no data, interpolate or use overall mean
        if np.any(bin_counts == 0):
            overall_mean = np.mean(V_current)
            valid_mask = bin_counts > 0
            if np.sum(valid_mask) >= 2:
                valid_indices = np.where(valid_mask)[0]
                valid_values = bin_means[valid_mask]
                bin_means = np.interp(np.arange(n_bins), valid_indices, valid_values)
            else:
                bin_means = np.where(bin_counts > 0, bin_means, overall_mean)

        # Apply light smoothing to reduce noise in continuation value estimates
        # Use a simple moving average with small window
        if n_bins > 4:
            kernel_size = 3
            kernel = np.ones(kernel_size) / kernel_size
            # Pad to handle edges
            padded = np.concatenate([bin_means[:1], bin_means, bin_means[-1:]])
            smoothed = np.convolve(padded, kernel, mode='valid')
            bin_means = smoothed

        cont_per_date[i] = bin_means.copy()

        # Get continuation value for each path
        cont_value = bin_means[bin_indices]
        continue_value = disc * cont_value

        # Exercise payoff
        exercise_payoff = np.maximum(K - S_i, 0.0)

        # Optimal value
        V_i = np.maximum(exercise_payoff, continue_value)

        # Update V_current for next iteration
        V_current = V_i

    # Build the policy object
    class ExercisePolicy:
        def __init__(self, bin_edges, cont_per_date, K, disc, n_dates):
            self.bin_edges = bin_edges
            self.cont_per_date = cont_per_date
            self.K = K
            self.disc = disc
            self.n_dates = n_dates
            self.n_bins = len(bin_edges) - 1

        def exercise(self, current_state, time):
            """
            Determine exercise decision for given states and time.
            
            Args:
                current_state: array of spot prices
                time: exercise date index (0 to n_dates-1)
            
            Returns:
                Boolean array: True if exercise
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            n = len(current_state)

            if time >= self.n_dates - 1:
                # Last exercise date: exercise if in-the-money
                return current_state < self.K

            # Bin the states
            bin_indices = np.digitize(current_state, self.bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

            # Get continuation value for each state
            cont_values = self.cont_per_date[time][bin_indices]
            continue_value = self.disc * cont_values

            # Exercise payoff
            exercise_payoff = np.maximum(self.K - current_state, 0.0)

            # Exercise if exercise payoff >= continuation value
            return exercise_payoff >= continue_value

    return ExercisePolicy(bin_edges, cont_per_date, K, disc, n_dates)
# EVOLVE-BLOCK-END
