"""Bermudan put exercise-policy research: Binned Backward Induction with Savitzky-Golay Smoothing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import savgol_filter

def fit_exercise_policy(training_paths, contract):
    """
    Fit an exercise policy for a Bermudan put option using binned backward induction 
    with Savitzky-Golay smoothing of continuation values to reduce Monte Carlo noise 
    and improve the accuracy of the exercise boundary.
    """
    start_time = time.time()
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    n_paths, n_cols = training_paths.shape
    
    # Determine exercise indices
    # Assume the last n_dates columns are the exercise dates
    start_idx = n_cols - n_dates
    
    dt = T / n_dates
    disc = np.exp(-r * dt)
    
    # Binning setup
    all_s = training_paths[:, start_idx:]
    s_min = np.min(all_s)
    s_max = np.max(all_s)
    
    # Increase bins to 400 for higher resolution
    # 400 bins is fast enough for 4096 paths and 12 steps
    n_bins = 400
    s_min_safe = max(s_min, 1e-6)
    s_max_safe = s_max * 1.01 if s_max > 0 else 1.0
    
    log_s_min = np.log(s_min_safe)
    log_s_max = np.log(s_max_safe)
    
    # Create log-spaced bin edges for better resolution at low prices
    log_edges = np.linspace(log_s_min, log_s_max, n_bins + 1)
    s_edges = np.exp(log_edges)
    
    # Calculate bin centers for interpolation
    bin_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    
    # Payoff at last date
    s_last = training_paths[:, n_cols - 1]
    payoff_last = np.maximum(K - s_last, 0.0)
    V = payoff_last.copy()
    
    # Store continuation values per date
    cont_vals_per_date = []
    
    # Backward induction
    for t_rel in range(n_dates - 2, -1, -1):
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
        cont_vals = np.zeros(n_bins)
        cont_vals[mask] = sum_v[mask] / counts[mask]
        
        # Fill empty bins using linear interpolation from nearest non-empty neighbors
        # This is a preliminary fill to allow the smoother to work on a complete series
        empty_mask = ~mask
        if np.any(empty_mask):
            non_empty_idx = np.where(mask)[0]
            if len(non_empty_idx) > 1:
                empty_idx = np.where(empty_mask)[0]
                # Interpolate empty bins
                # We can use np.interp on the indices for efficiency
                # But since the grid is uniform in log-space (mostly), simple linear interp on bin index is okay for filling
                # However, we want to preserve the shape.
                # Let's just use the nearest neighbor or linear interp on the non-empty points
                interp_vals = np.interp(np.arange(n_bins), non_empty_idx, cont_vals[non_empty_idx])
                cont_vals[empty_mask] = interp_vals[empty_mask]
            elif len(non_empty_idx) == 1:
                cont_vals[empty_mask] = cont_vals[non_empty_idx[0]]
        
        # Apply Savitzky-Golay filter to smooth the continuation values
        # This reduces Monte Carlo noise while preserving the curvature of the value function.
        # Window length must be odd. Poly order must be less than window length.
        # We want a smooth but responsive filter.
        # For n_bins=400, a window of 15-31 is reasonable.
        # Let's use a window size that is a factor of the bin count or just a fixed odd number.
        # 21 is a good balance.
        window_size = 21
        poly_order = 3
        
        # Ensure window size is valid
        if n_bins > window_size:
            # The filter requires the data to be at least window_size long
            # It also handles the boundaries by extending the signal.
            try:
                cont_vals_smoothed = savgol_filter(cont_vals, window_size, poly_order)
            except Exception:
                # Fallback to simple moving average if savgol fails (unlikely)
                kernel = np.ones(window_size) / window_size
                cont_vals_smoothed = np.convolve(cont_vals, kernel, mode='same')
        else:
            cont_vals_smoothed = cont_vals
        
        # The smoothed values might go slightly negative due to numerical issues, clip them
        cont_vals_smoothed = np.maximum(cont_vals_smoothed, 0.0)
        
        cont_vals_per_date.append(cont_vals_smoothed)
        
        # Calculate new V
        payoff_t = np.maximum(K - s_t, 0.0)
        
        # Use the smoothed continuation values for the next step's binning
        # This propagates the smoothing effect backward
        cont_path = cont_vals_smoothed[bin_idx]
        disc_cont = disc * cont_path
        V = np.maximum(payoff_t, disc_cont)
        
        # Time check
        if time.time() - start_time > 130:
            break
    
    # Reverse to match time order
    cont_vals_per_date.reverse()
    
    # If we broke out early, we might have missing dates. 
    # For simplicity, if we don't have all dates, we fill the rest with the last known or a heuristic.
    # However, 12 steps with 400 bins is very fast (< 1s usually).
    # Let's ensure we have n_dates - 1 entries (since the last date is just payoff, no continuation needed for exercise decision at T)
    # Wait, the loop runs from n_dates-2 down to 0. So it generates n_dates-1 continuation arrays.
    # ExercisePolicy expects cont_vals_per_date[t_rel] where t_rel is 0 to n_dates-2.
    # At t_rel = n_dates-1 (last date), we exercise if ITM.
    
    if len(cont_vals_per_date) < n_dates - 1:
        # Pad with zeros or the last value if we ran out of time
        while len(cont_vals_per_date) < n_dates - 1:
            cont_vals_per_date.append(np.zeros(n_bins))
            
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
            if time < 1:
                return np.zeros(len(current_state), dtype=bool)
            
            t_rel = int(time) - 1
            
            if t_rel < 0 or t_rel >= self.n_dates:
                return np.zeros(len(current_state), dtype=bool)
            
            if t_rel == self.n_dates - 1:
                return current_state < self.K
            
            # Get continuation values
            cont_vals = self.cont_vals[t_rel]
            
            # Linear interpolation on bin centers
            # np.interp is fast and handles the interpolation cleanly
            disc_cont = self.disc * np.interp(current_state, self.bin_centers, cont_vals)
            
            payoff = np.maximum(self.K - current_state, 0.0)
            
            return payoff >= disc_cont
    
    return ExercisePolicy(K, disc, s_edges, bin_centers, n_bins, cont_vals_per_date, n_dates)

# EVOLVE-BLOCK-END
