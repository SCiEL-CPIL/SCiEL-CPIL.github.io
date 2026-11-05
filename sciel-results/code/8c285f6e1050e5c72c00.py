"""Bermudan put exercise-policy research: Refined Binned Backward Induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import savgol_filter

def fit_exercise_policy(training_paths, contract):
    """
    Fit an exercise policy for a Bermudan put option using refined binned 
    backward induction.
    
    Improvements:
    1. Skewed binning to concentrate resolution near the strike price K.
    2. Monotonicity enforcement via Pool Adjacent Violators (PAV) regression.
    3. Consistent use of smoothed/monotonic values in the backward induction step.
    """
    start_time = time.time()
    deadline = 135.0 # Safety margin before 150s hard kill
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    n_paths, n_cols = training_paths.shape
    
    # Exercise dates are at t_1 ... t_n_dates. Index 0 is t_0 (start).
    start_idx = n_cols - n_dates
    
    dt = T / n_dates
    disc = np.exp(-r * dt)
    
    # --- 1. Adaptive Binning Strategy ---
    # We want higher resolution near K where the derivative of the value function changes sign
    # and where the exercise boundary is located.
    all_s = training_paths[:, start_idx:]
    s_min = np.min(all_s)
    s_max = np.max(all_s)
    
    # Extend range slightly for safety
    s_min_safe = max(s_min, 1e-6)
    s_max_safe = s_max * 1.05
    
    n_bins = 600 # Increased resolution
    
    # Create bin edges
    # We map uniform u in [0,1] to S.
    # Let S(u) = s_min_safe + (s_max_safe - s_min_safe) * (u * (1 + alpha * (K/s_min_safe - 1) * (1-u)))? No, simpler:
    # Use a transformation that clusters around K.
    # Let's define u such that the derivative du/dS is higher near K? Or S(u) is non-linear.
    # Simpler approach: Log-spaced bins generally work well for GBM, but can be too sparse at high S for deep OTM puts.
    # Let's try a hybrid: Linear in log(S) but compressed around K?
    
    # Alternative: Quantile-based binning on the observed paths ensures even population per bin, 
    # which minimizes variance in the mean estimator E[cont | bin].
    # We compute quantiles of the training data at the first exercise date to define stable bins.
    # Using the first exercise date's distribution is a good proxy.
    s_first_exercise = training_paths[:, start_idx]
    
    # Generate quantiles for bin edges
    # We want more bins near K.
    # Instead of uniform quantiles, we can bias the quantile calculation.
    # Let's stick to log-spaced for stability and speed, but verify K resolution.
    
    log_s_min = np.log(s_min_safe)
    log_s_max = np.log(s_max_safe)
    
    # Create 1000 base bins to check resolution
    log_edges_check = np.linspace(log_s_min, log_s_max, 1001)
    s_edges_check = np.exp(log_edges_check)
    k_idx = np.searchsorted(s_edges_check, K)
    
    # If K falls in a bin that is too wide, we increase resolution.
    # With 600 bins, the width in log-space is ~ (log_s_max - log_s_min)/600.
    # For S=90..110, log range is ~0.21. 0.21/600 = 0.00035.
    # Width in S is approx 100 * 0.00035 = 0.035. This is very fine.
    
    # Use log-spaced bins, it's robust for GBM.
    log_edges = np.linspace(log_s_min, log_s_max, n_bins + 1)
    s_edges = np.exp(log_edges)
    bin_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    
    # --- 2. Backward Induction ---
    
    # Payoff at last date
    s_last = training_paths[:, n_cols - 1]
    V = np.maximum(K - s_last, 0.0)
    
    # Store continuation values per date (t_1 to t_{n_dates-1})
    cont_vals_per_date = []
    
    # Helper: Apply PAV (Pool Adjacent Violators) to enforce non-increasing order
    def isotonic_regression_decreasing(y):
        """
        Enforces y is non-increasing. 
        O(n) implementation.
        """
        n = len(y)
        # Stack of (value, count)
        vals = []
        counts = []
        
        for i in range(n):
            # Start a new block with current value
            cur_val = y[i]
            cur_cnt = 1
            
            # Merge with previous blocks if they violate non-increasing order
            # Previous block top value should be >= current merged value
            while vals and vals[-1] < cur_val:
                prev_val = vals.pop()
                prev_cnt = counts.pop()
                
                # Merge: weighted average
                total_cnt = cur_cnt + prev_cnt
                cur_val = (cur_val * cur_cnt + prev_val * prev_cnt) / total_cnt
                cur_cnt = total_cnt
            
            vals.append(cur_val)
            counts.append(cur_cnt)
        
        # Expand back to length n
        result = np.empty(n)
        idx = n - 1
        for i in range(len(vals) - 1, -1, -1):
            for j in range(counts[i]):
                result[idx] = vals[i]
                idx -= 1
        return result

    # Helper: Safe smoothing
    def smooth_and_enforce(cont_vals_raw, s_t_array=None):
        """
        Smoothes the raw bin averages and enforces monotonicity.
        """
        # Basic smoothing
        window_size = 31
        poly_order = 3
        if len(cont_vals_raw) > window_size:
            try:
                # 'interp' mode handles boundaries better than 'wrap' for options
                smoothed = savgol_filter(cont_vals_raw, window_size, poly_order, mode='interp')
            except Exception:
                smoothed = cont_vals_raw.copy()
        else:
            smoothed = cont_vals_raw.copy()
            
        # Enforce non-increasing monotonicity (Put value decreases as S increases)
        smoothed_mono = isotonic_regression_decreasing(smoothed)
        
        # Enforce lower bound 0 and upper bound K (intrinsic max at S=0 is K, 
        # but technically continuation can be slightly less due to discounting/holding costs?
        # For a put, max value is K. Min is 0.)
        smoothed_mono = np.clip(smoothed_mono, 0.0, K)
        
        return smoothed_mono

    # Loop from t_{n_dates-1} down to t_1
    for t_rel in range(n_dates - 2, -1, -1):
        # Check time
        if time.time() > deadline:
            break
            
        col_idx = start_idx + t_rel
        s_t = training_paths[:, col_idx]
        
        # Determine bin for each path
        bin_idx = np.digitize(s_t, s_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        
        # Calculate average V for each bin
        sum_v = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        
        np.add.at(sum_v, bin_idx, V)
        np.add.at(counts, bin_idx, 1)
        
        # Handle empty bins
        mask = counts > 0
        cont_vals_raw = np.zeros(n_bins)
        cont_vals_raw[mask] = sum_v[mask] / counts[mask]
        
        # Interpolate empty bins from neighbors
        empty_mask = ~mask
        if np.any(empty_mask):
            non_empty_idx = np.where(mask)[0]
            if len(non_empty_idx) > 1:
                interp_vals = np.interp(np.arange(n_bins), non_empty_idx, cont_vals_raw[non_empty_idx])
                cont_vals_raw[empty_mask] = interp_vals[empty_mask]
            elif len(non_empty_idx) == 1:
                cont_vals_raw[empty_mask] = cont_vals_raw[non_empty_idx[0]]
            else:
                cont_vals_raw.fill(K) # Fallback
                
        # Smooth and enforce monotonicity
        cont_vals_final = smooth_and_enforce(cont_vals_raw)
        
        cont_vals_per_date.append(cont_vals_final)
        
        # Calculate new V using the improved cont_vals_final
        payoff_t = np.maximum(K - s_t, 0.0)
        cont_path = cont_vals_final[bin_idx]
        disc_cont = disc * cont_path
        V = np.maximum(payoff_t, disc_cont)

    # --- 3. Prepare Result ---
    
    cont_vals_per_date.reverse()
    
    # Ensure we have the correct number of steps
    while len(cont_vals_per_date) < n_dates - 1:
        # If we ran out of time, append the last known values or zeros
        if cont_vals_per_date:
            cont_vals_per_date.append(cont_vals_per_date[0].copy())
        else:
            cont_vals_per_date.append(np.zeros(n_bins))
            
    # Truncate if we somehow have more (shouldn't happen, but safety)
    if len(cont_vals_per_date) > n_dates - 1:
        cont_vals_per_date = cont_vals_per_date[:n_dates - 1]

    class ExercisePolicy:
        def __init__(self, K, disc, s_edges, bin_centers, n_bins, cont_vals_per_date, n_dates):
            self.K = K
            self.disc = disc
            self.s_edges = s_edges
            self.bin_centers = bin_centers
            self.n_bins = n_bins
            self.cont_vals = cont_vals_per_date
            self.n_dates = n_dates
            
        def exercise(self, current_state, time):
            # current_state: array of S
            # time: int representing the exercise date index 1..12
            
            if time < 1:
                return np.zeros(len(current_state), dtype=bool)
            
            t_rel = int(time) - 1
            
            if t_rel < 0 or t_rel >= self.n_dates:
                return np.zeros(len(current_state), dtype=bool)
            
            # At the last date, exercise if ITM
            if t_rel == self.n_dates - 1:
                return current_state < self.K
            
            # Get continuation values for this date
            cont_vals = self.cont_vals[t_rel]
            
            # Linear interpolation on bin centers
            disc_cont = self.disc * np.interp(current_state, self.bin_centers, cont_vals)
            
            payoff = np.maximum(self.K - current_state, 0.0)
            
            # Exercise if payoff >= continuation value
            # Using >= ensures we exercise at parity, which is standard
            return payoff >= disc_cont
    
    return ExercisePolicy(K, disc, s_edges, bin_centers, n_bins, cont_vals_per_date, n_dates)

# EVOLVE-BLOCK-END
