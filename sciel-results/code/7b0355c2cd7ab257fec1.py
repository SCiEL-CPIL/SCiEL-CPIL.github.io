"""Bermudan put exercise-policy research: Binned Backward Induction with Linear Interpolation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit an exercise policy for a Bermudan put option using binned backward induction 
    with linear interpolation of continuation values to reduce bias.
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
    
    # Increase bins to 200 for better resolution
    n_bins = 200
    s_min = max(s_min, 1e-6)
    log_s_min = np.log(s_min)
    log_s_max = np.log(s_max)
    
    # Create log-spaced bin edges
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
        
        # For empty bins, use linear interpolation from nearest non-empty neighbors
        empty_mask = ~mask
        if np.any(empty_mask):
            non_empty_idx = np.where(mask)[0]
            if len(non_empty_idx) > 1:
                empty_idx = np.where(empty_mask)[0]
                # For each empty bin, find the two nearest non-empty bins and interpolate
                for ei in empty_idx:
                    # Find indices of non-empty bins
                    left_mask = non_empty_idx < ei
                    right_mask = non_empty_idx > ei
                    
                    if np.any(left_mask) and np.any(right_mask):
                        left_nearest = non_empty_idx[left_mask][-1]
                        right_nearest = non_empty_idx[right_mask][0]
                        # Linear interpolation
                        w = (ei - left_nearest) / (right_nearest - left_nearest)
                        cont_vals[ei] = (1 - w) * cont_vals[left_nearest] + w * cont_vals[right_nearest]
                    elif np.any(left_mask):
                        cont_vals[ei] = cont_vals[non_empty_idx[left_mask][-1]]
                    elif np.any(right_mask):
                        cont_vals[ei] = cont_vals[non_empty_idx[right_mask][0]]
            elif len(non_empty_idx) == 1:
                cont_vals[empty_mask] = cont_vals[non_empty_idx[0]]
        
        cont_vals_per_date.append(cont_vals)
        
        # Calculate new V
        payoff_t = np.maximum(K - s_t, 0.0)
        
        # For the backward induction step, we need the continuation value for each path.
        # Using the binned average is standard, but we can use the interpolated value
        # if we want to be more consistent. However, for the *training* step, 
        # using the bin average is sufficient and faster. 
        # The improvement comes from the *evaluation* step where we interpolate.
        
        cont_path = cont_vals[bin_idx]
        disc_cont = disc * cont_path
        V = np.maximum(payoff_t, disc_cont)
    
    # Reverse to match time order
    cont_vals_per_date.reverse()
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 140:
        # Return a simple policy if running out of time
        class SimplePolicy:
            def __init__(self, K):
                self.K = K
            def exercise(self, current_state, time):
                if time < 1:
                    return np.zeros(len(current_state), dtype=bool)
                return current_state < self.K
        return SimplePolicy(K)
    
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
            
            t_rel = time - 1
            
            if t_rel < 0 or t_rel >= self.n_dates:
                return np.zeros(len(current_state), dtype=bool)
            
            if t_rel == self.n_dates - 1:
                return current_state < self.K
            
            # Get bin indices
            bin_idx = np.digitize(current_state, self.s_edges) - 1
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            
            # Get continuation values
            cont_vals = self.cont_vals[t_rel]
            
            # Linear interpolation for better accuracy
            # If current_state is within the bin, interpolate between left and right edge centers?
            # Or simply use the bin value? 
            # Interpolating between bin centers is better.
            # Find the bin index b. The value is approximated by interpolating 
            # between cont_vals[b-1] and cont_vals[b] if S is closer to left, etc.
            # Actually, simpler: use np.interp on the bin_centers.
            
            # np.interp is fast and handles the interpolation cleanly
            disc_cont = self.disc * np.interp(current_state, self.bin_centers, cont_vals)
            
            payoff = np.maximum(self.K - current_state, 0.0)
            
            return payoff >= disc_cont
    
    return ExercisePolicy(K, disc, s_edges, bin_centers, n_bins, cont_vals_per_date, n_dates)

# EVOLVE-BLOCK-END
