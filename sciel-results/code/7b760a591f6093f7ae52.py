"""Bermudan put exercise-policy research with monotonic continuation value estimation."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

class ExercisePolicy:
    def __init__(self, thresholds, r, dt):
        self.thresholds = thresholds
        self.r = r
        self.dt = dt

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
    deadline = start_time + 110.0  # Safety margin

    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    T = contract.get('T', 1.0)
    
    # Determine shape of training_paths
    n_dates, n_paths = training_paths.shape
    dt = T / n_dates
    discount_factor = np.exp(-r * dt)

    # Initialize values
    V = np.zeros((n_dates, n_paths))
    
    # Terminal payoff
    V[-1] = np.maximum(K - training_paths[-1], 0.0)
    
    # Store monotonic continuation value tables for each time step
    # Each entry will be (bin_centers, monotonic_bin_values)
    cont_tables = [None] * n_dates
    
    # Parameters for binning
    # Using a moderate number of bins to balance noise reduction and resolution
    n_bins = 100 

    # Backward Induction
    for t in range(n_dates - 2, -1, -1):
        if _time.time() > deadline - 15.0:
            # Fallback: if running out of time, use a simple heuristic for remaining steps
            # For a put, threshold increases as time to maturity decreases.
            # Approximate threshold: K * (1 - exp(-r * time_to_maturity)) is not quite right.
            # Simple heuristic: Threshold is lower earlier.
            for t_rem in range(t, -1, -1):
                time_to_maturity = (n_dates - 1 - t_rem) * dt
                # Rough approximation: threshold is around 0.5K to 0.9K
                # Earlier dates -> lower threshold
                frac = (n_dates - 1 - t_rem) / n_dates
                approx_thresh = K * (0.5 + 0.4 * (1 - frac))
                # Create a dummy table that implies this threshold
                # We need a function m(S) such that K-S = d*m(S) at S=approx_thresh
                # This is complex to fake. Just use a flat continuation value?
                # If we assume no exercise (continuation = 0), threshold is K.
                # If we assume immediate exercise, threshold is 0.
                # Let's just set thresholds directly in the final step if we break here.
                cont_tables[t_rem] = None
            break
        
        S_t = training_paths[t]
        V_next = V[t + 1]
        
        # 1. Sort S_t
        sorted_indices = np.argsort(S_t)
        sorted_S = S_t[sorted_indices]
        sorted_V_next = V_next[sorted_indices]
        
        # 2. Create bins
        # Use quantile binning to ensure equal population, but for monotonicity,
        # fixed width might be better if S is well-distributed?
        # Quantile binning is safer for tails.
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(sorted_S, quantiles)
        bin_edges = np.unique(bin_edges)
        
        if len(bin_edges) < 2:
            # Degenerate case
            cont_val = discount_factor * np.mean(V_next)
            V[t] = np.maximum(np.maximum(K - S_t, 0.0), cont_val)
            cont_tables[t] = (np.array([0.0, 1.0]), np.array([np.mean(V_next), np.mean(V_next)]))
            continue
        
        # Assign bins
        bin_indices = np.digitize(sorted_S, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
        
        # 3. Compute bin means
        n_actual_bins = len(bin_edges) - 1
        bin_sums = np.zeros(n_actual_bins)
        bin_counts = np.zeros(n_actual_bins)
        
        np.add.at(bin_sums, bin_indices, sorted_V_next)
        np.add.at(bin_counts, bin_indices, 1)
        
        valid_bins = bin_counts > 0
        bin_means = np.zeros(n_actual_bins)
        bin_means[valid_bins] = bin_sums[valid_bins] / bin_counts[valid_bins]
        
        # Fill empty bins via interpolation first to have a complete array
        valid_bin_indices = np.where(valid_bins)[0]
        if len(valid_bin_indices) > 0 and len(valid_bin_indices) < n_actual_bins:
            invalid_bins = np.where(~valid_bins)[0]
            bin_means[invalid_bins] = np.interp(invalid_bins, valid_bin_indices, bin_means[valid_bin_indices])
        elif len(valid_bin_indices) == 0:
            bin_means[:] = np.mean(V_next)
            
        # 4. Enforce Monotonicity
        # For a put, as S increases, E[V_next | S] should decrease (or stay same).
        # So bin_means should be non-increasing.
        # We enforce this by taking the running minimum from the right (highest S) to the left.
        # Wait, if S is sorted ascending, index 0 is low S, index n-1 is high S.
        # Low S -> High V. High S -> Low V.
        # So bin_means[0] >= bin_means[1] >= ... >= bin_means[n-1].
        # We iterate from right to left:
        # m[i] = min(bin_means[i], m[i+1])
        monotonic_means = np.zeros(n_actual_bins)
        monotonic_means[-1] = bin_means[-1]
        for i in range(n_actual_bins - 2, -1, -1):
            monotonic_means[i] = min(bin_means[i], monotonic_means[i + 1])
            
        # 5. Define bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        # Store the monotonic table for threshold extraction
        cont_tables[t] = (bin_centers, monotonic_means)
        
        # 6. Interpolate back to original S_t
        # We use the monotonic means for interpolation
        cont_values_sorted = np.interp(sorted_S, bin_centers, monotonic_means)
        
        # Map back to original order
        cont_values = np.zeros(n_paths)
        cont_values[sorted_indices] = cont_values_sorted
        
        # Discount
        cont_values *= discount_factor
        
        # Payoff if exercised now
        payoff_now = np.maximum(K - S_t, 0.0)
        
        # Optimal value
        V[t] = np.maximum(payoff_now, cont_values)
        
    # Determine thresholds by reconstructing the value function on a fine grid
    thresholds = np.zeros(n_dates)
    
    # Fine grid for threshold search
    # Stock prices for a put threshold are in [0, K]
    grid_S = np.linspace(0, K, 500)
    
    for t in range(n_dates):
        if _time.time() > deadline - 5.0:
             thresholds[t] = K * 0.8
             continue

        if t == n_dates - 1:
            # At maturity, exercise if S < K
            thresholds[t] = K
            continue
            
        if cont_tables[t] is None:
            # Fallback if backward induction was cut short
            # Use a heuristic based on time
            time_frac = (n_dates - 1 - t) / n_dates
            thresholds[t] = K * (0.5 + 0.4 * time_frac)
            continue
            
        bin_centers, bin_values = cont_tables[t]
        
        # Interpolate the continuation value onto the fine grid
        # bin_values are E[V_{t+1}], so we must discount them to get C_t(S)
        cont_grid = np.interp(grid_S, bin_centers, bin_values) * discount_factor
        
        # Payoff if exercised now
        payoff_grid = np.maximum(K - grid_S, 0.0)
        
        # Difference: Payoff - Continuation
        diff = payoff_grid - cont_grid
        
        # Find the largest S where diff > 0
        # The exercise region is S < threshold.
        # Since grid_S is sorted ascending, we look for the last index where diff > 0.
        exercise_mask = diff > 1e-8
        
        if np.any(exercise_mask):
            last_exercise_idx = np.where(exercise_mask)[0][-1]
            
            # Interpolate the threshold
            if last_exercise_idx < len(grid_S) - 1:
                S1 = grid_S[last_exercise_idx]
                S2 = grid_S[last_exercise_idx + 1]
                D1 = diff[last_exercise_idx]
                D2 = diff[last_exercise_idx + 1]
                
                if D1 != D2:
                    thresholds[t] = S1 + (0 - D1) * (S2 - S1) / (D2 - D1)
                else:
                    thresholds[t] = S1
            else:
                thresholds[t] = grid_S[-1]
        else:
            # No exercise region found?
            # This can happen if continuation value is very high even at S=0?
            # Unlikely for a put.
            thresholds[t] = 0.0
            
        # Clip to [0, K]
        thresholds[t] = np.clip(thresholds[t], 0.0, K)

    return ExercisePolicy(thresholds, r, dt)

# EVOLVE-BLOCK-END
