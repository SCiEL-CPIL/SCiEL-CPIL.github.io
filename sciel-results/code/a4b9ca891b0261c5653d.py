"""Bermudan put exercise-policy research: Binned Backward Induction with Adaptive Smoothing and Isotonic Monotonicity."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression

def fit_exercise_policy(training_paths, contract):
    """
    Fit an exercise policy for a Bermudan put option using binned backward induction 
    with adaptive Savitzky-Golay smoothing, isotonic regression for monotonicity,
    and PCHIP interpolation.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    n_paths, n_cols = training_paths.shape
    
    # Determine exercise indices
    start_idx = n_cols - n_dates
    
    dt = T / n_dates
    disc = np.exp(-r * dt)
    
    # Binning setup - use quantile-based bins for better distribution
    all_s = training_paths[:, start_idx:]
    s_min = np.min(all_s)
    s_max = np.max(all_s)
    
    # Use 1500 bins - good balance of resolution and computation
    n_bins = 1500
    s_min_safe = max(s_min, 1e-6)
    s_max_safe = s_max * 1.02 if s_max > 0 else 1.0
    
    # Log-spaced bin edges
    log_s_min = np.log(s_min_safe)
    log_s_max = np.log(s_max_safe)
    log_edges = np.linspace(log_s_min, log_s_max, n_bins + 1)
    s_edges = np.exp(log_edges)
    
    # Bin centers
    bin_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    
    # Payoff at last date
    s_last = training_paths[:, n_cols - 1]
    payoff_last = np.maximum(K - s_last, 0.0)
    V = payoff_last.copy()
    
    # Store interpolants per date
    interp_per_date = []
    
    # Backward induction
    for t_rel in range(n_dates - 2, -1, -1):
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
        cont_vals = np.zeros(n_bins)
        cont_vals[mask] = sum_v[mask] / counts[mask]
        
        # Fill empty bins using interpolation from nearest non-empty neighbors
        empty_mask = ~mask
        if np.any(empty_mask):
            non_empty_idx = np.where(mask)[0]
            if len(non_empty_idx) > 1:
                interp_vals = np.interp(np.arange(n_bins), non_empty_idx, cont_vals[non_empty_idx])
                cont_vals[empty_mask] = interp_vals[empty_mask]
            elif len(non_empty_idx) == 1:
                cont_vals[empty_mask] = cont_vals[non_empty_idx[0]]
        
        # Adaptive Savitzky-Golay smoothing
        # Use smaller window for earlier dates (less data, more noise relative to signal)
        # and larger window for later dates (more data, smoother)
        progress = (n_dates - 2 - t_rel) / max(n_dates - 2, 1)  # 0 at last, 1 at first
        window_size = int(21 + 20 * (1 - progress))  # 41 to 21
        window_size = min(window_size, n_bins - 1)
        if window_size % 2 == 0:
            window_size += 1
        window_size = max(window_size, 5)
        poly_order = 3
        
        if n_bins > window_size:
            try:
                cont_vals_smoothed = savgol_filter(cont_vals, window_size, poly_order)
            except Exception:
                kernel = np.ones(window_size) / window_size
                cont_vals_smoothed = np.convolve(cont_vals, kernel, mode='same')
        else:
            cont_vals_smoothed = cont_vals.copy()
        
        # Enforce monotonicity using isotonic regression (non-increasing for puts)
        # Invert for isotonic_regression which expects non-decreasing
        inverted = -cont_vals_smoothed
        try:
            monotonic_inverted = isotonic_regression(inverted)
            cont_vals_smoothed = -monotonic_inverted
        except Exception:
            # Fallback: simple forward-backward pass
            # Forward pass (non-increasing)
            for i in range(1, n_bins):
                if cont_vals_smoothed[i] > cont_vals_smoothed[i-1]:
                    cont_vals_smoothed[i] = cont_vals_smoothed[i-1]
            # Backward pass (non-increasing)
            for i in range(n_bins-2, -1, -1):
                if cont_vals_smoothed[i] > cont_vals_smoothed[i+1]:
                    cont_vals_smoothed[i] = cont_vals_smoothed[i+1]
        
        # Ensure non-negative
        cont_vals_smoothed = np.maximum(cont_vals_smoothed, 0.0)
        
        # Create PCHIP interpolant
        try:
            interp = PchipInterpolator(bin_centers, cont_vals_smoothed)
        except Exception:
            interp = PchipInterpolator(bin_centers, np.maximum(cont_vals_smoothed, 0))
            
        interp_per_date.append(interp)
        
        # Calculate new V
        payoff_t = np.maximum(K - s_t, 0.0)
        
        # Evaluate continuation value using the interpolant
        cont_path = interp(s_t)
        disc_cont = disc * cont_path
        V = np.maximum(payoff_t, disc_cont)
    
    # Reverse to match time order
    interp_per_date.reverse()
    
    if len(interp_per_date) < n_dates - 1:
        # Pad with last available or zero interpolants if time ran out
        if len(interp_per_date) > 0:
            last_interp = interp_per_date[-1]
        else:
            last_interp = PchipInterpolator(bin_centers, np.zeros(n_bins))
        while len(interp_per_date) < n_dates - 1:
            interp_per_date.append(last_interp)
            
    class ExercisePolicy:
        def __init__(self, K, disc, interp_per_date, n_dates):
            self.K = K
            self.disc = disc
            self.interps = interp_per_date
            self.n_dates = n_dates
            
        def exercise(self, current_state, time):
            if time < 1:
                return np.zeros(len(current_state), dtype=bool)
            
            t_rel = int(time) - 1
            
            if t_rel < 0 or t_rel >= self.n_dates:
                return np.zeros(len(current_state), dtype=bool)
            
            if t_rel == self.n_dates - 1:
                return current_state < self.K
            
            # Get continuation value
            interp = self.interps[t_rel]
            cont_vals = interp(current_state)
            disc_cont = self.disc * cont_vals
            
            payoff = np.maximum(self.K - current_state, 0.0)
            
            return payoff >= disc_cont
    
    return ExercisePolicy(K, disc, interp_per_date, n_dates)

# EVOLVE-BLOCK-END
