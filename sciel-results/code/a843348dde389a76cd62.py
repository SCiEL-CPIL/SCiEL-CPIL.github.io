# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy using a parametric approach with
    a cubic polynomial basis in log-price for continuation values,
    combined with a local refinement step using nearest-neighbor
    regression on the training paths.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Extract parameters
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)

    # Handle input shape
    n_paths, n_cols = training_paths.shape
    if n_cols == n_dates:
        exercise_data = training_paths
    elif n_cols == n_dates + 1:
        exercise_data = training_paths[:, 1:n_dates+1]
    else:
        exercise_data = training_paths[:, :n_dates]

    # We will use a hybrid approach:
    # 1. Global cubic polynomial regression in log-price for the continuation value
    # 2. Local refinement: for each time step, we also compute a local average
    #    of the next-step values weighted by proximity in log-space

    # Features for cubic polynomial: [1, ln(S), (ln(S))^2, (ln(S))^3]
    def get_features_cubic(S):
        log_S = np.log(S)
        return np.column_stack([np.ones_like(log_S), log_S, log_S**2, log_S**3])

    # Initialize V for the last date
    V_vals = [None] * n_dates
    S_vals = [exercise_data[:, t] for t in range(n_dates)]
    log_S_vals = [np.log(S_vals[t]) for t in range(n_dates)]

    # Last date: V = Payoff
    V_vals[-1] = np.maximum(K - S_vals[-1], 0.0)

    # Store coefficients for continuation value function
    # We'll store both global cubic coeffs and local weights
    cont_coeffs_global = [None] * n_dates
    cont_coeffs_local = [None] * n_dates  # For local averaging

    dt = T / n_dates
    discount = np.exp(-r * dt)

    for t in range(n_dates - 2, -1, -1):
        if time.time() > deadline:
            break

        S_t = S_vals[t]
        log_S_t = log_S_vals[t]
        V_next_vals = V_vals[t+1]

        # Method 1: Global cubic polynomial regression
        X_global = get_features_cubic(S_t)
        try:
            coeffs_global, _, _, _ = np.linalg.lstsq(X_global, V_next_vals, rcond=None)
        except np.linalg.LinAlgError:
            coeffs_global = np.array([np.mean(V_next_vals), 0.0, 0.0, 0.0])

        cont_coeffs_global[t] = coeffs_global

        # Method 2: Local refinement using nearest neighbors in log-space
        # For each path at time t, find the nearest paths at time t+1 in log-space
        # and compute a weighted average of V_next_vals
        # This is expensive, so we'll use a grid-based approach for efficiency

        # Create a grid of log-S values at time t+1
        n_grid = 200
        log_S_min = np.min(log_S_vals[t+1])
        log_S_max = np.max(log_S_vals[t+1])
        log_S_grid = np.linspace(log_S_min, log_S_max, n_grid)

        # Bin the V_next_vals by log_S at time t+1
        bin_indices = np.digitize(log_S_vals[t+1], log_S_grid) - 1
        bin_indices = np.clip(bin_indices, 0, n_grid - 1)

        # Compute bin averages
        bin_sums = np.zeros(n_grid)
        bin_counts = np.zeros(n_grid)
        np.add.at(bin_sums, bin_indices, V_next_vals)
        np.add.at(bin_counts, bin_indices, 1)

        # Handle empty bins
        valid_bins = bin_counts > 0
        bin_avgs = np.zeros(n_grid)
        bin_avgs[valid_bins] = bin_sums[valid_bins] / bin_counts[valid_bins]

        # For empty bins, interpolate from neighbors
        if not np.all(valid_bins):
            # Simple linear interpolation for empty bins
            for i in range(n_grid):
                if not valid_bins[i]:
                    # Find nearest valid bins
                    left = i - 1
                    right = i + 1
                    while left >= 0 and not valid_bins[left]:
                        left -= 1
                    while right < n_grid and not valid_bins[right]:
                        right += 1
                    if left >= 0 and right < n_grid:
                        # Linear interpolation
                        w = (i - left) / (right - left)
                        bin_avgs[i] = (1 - w) * bin_avgs[left] + w * bin_avgs[right]
                    elif left >= 0:
                        bin_avgs[i] = bin_avgs[left]
                    elif right < n_grid:
                        bin_avgs[i] = bin_avgs[right]
                    else:
                        bin_avgs[i] = np.mean(V_next_vals)

        # Interpolate bin averages to get values at S_t
        # Use linear interpolation
        cont_vals_local = np.interp(log_S_t, log_S_grid, bin_avgs)

        # Store local continuation values
        cont_coeffs_local[t] = (log_S_grid, bin_avgs)

        # Combine global and local estimates
        # Use a weighted average: 50% global, 50% local
        # The global model captures the smooth trend, the local model captures local variations
        cont_vals_global = X_global @ coeffs_global
        cont_vals_combined = 0.5 * cont_vals_global + 0.5 * cont_vals_local

        # Discount
        cont_vals_discounted = cont_vals_combined * discount

        # Intrinsic value
        intrinsic_vals = np.maximum(K - S_t, 0.0)

        # Total Value
        V_vals[t] = np.maximum(intrinsic_vals, cont_vals_discounted)

    # Construct the policy object
    class ExercisePolicy:
        def __init__(self, K, cont_coeffs_global, cont_coeffs_local, n_dates, r, T):
            self.K = K
            self.cont_coeffs_global = cont_coeffs_global
            self.cont_coeffs_local = cont_coeffs_local
            self.n_dates = n_dates
            self.r = r
            self.T = T
            self.dt = T / n_dates
            self.discount = np.exp(-self.r * self.dt)

        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: index of the exercise date (0 to n_dates-1)
            """
            # If time is the last date, always exercise if intrinsic > 0
            if time >= self.n_dates - 1:
                return (self.K - current_state) > 0

            if time < 0 or time >= len(self.cont_coeffs_global) or self.cont_coeffs_global[time] is None:
                # Fallback
                return (self.K - current_state) > 0

            coeffs = self.cont_coeffs_global[time]
            log_S = np.log(current_state)

            # Global model
            cont_val_global = (coeffs[0] + coeffs[1]*log_S + coeffs[2]*log_S**2 + coeffs[3]*log_S**3)

            # Local model
            if self.cont_coeffs_local[time] is not None:
                log_S_grid, bin_avgs = self.cont_coeffs_local[time]
                cont_val_local = np.interp(log_S, log_S_grid, bin_avgs)
            else:
                cont_val_local = cont_val_global

            # Combine
            cont_val = 0.5 * cont_val_global + 0.5 * cont_val_local
            cont_val_discounted = cont_val * self.discount

            intrinsic_val = np.maximum(self.K - current_state, 0.0)

            return intrinsic_val >= cont_val_discounted

    return ExercisePolicy(K, cont_coeffs_global, cont_coeffs_local, n_dates, r, T)

# EVOLVE-BLOCK-END
