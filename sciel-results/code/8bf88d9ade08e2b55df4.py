"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import UnivariateSpline

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    dt = T / n_dates
    discount = np.exp(-r * dt)
    
    n_paths = training_paths.shape[0]
    
    # S_all shape: (n_paths, 12) corresponding to dates 1..12
    # training_paths has n_dates+1 columns (0..12), we use 1..12
    S_all = training_paths[:, 1:n_dates+1]
    
    # V[i] will store the value of the option at exercise date i+1 (0-indexed i)
    # V[11] is the value at the last date (maturity)
    V = np.zeros((n_dates, n_paths))
    
    # At maturity (last date, index 11), the value is the European payoff
    V[11, :] = np.maximum(K - S_all[:, 11], 0.0)
    
    # We will store the spline objects for the continuation value function at each date
    splines = [None] * n_dates
    
    # Number of bins for the binned spline approach
    # Too many bins -> noisy; too few -> biased. 100-200 is a good range for N=4096.
    n_bins = 150
    
    # Backward induction
    for i in range(10, -1, -1):
        if time.time() > deadline:
            # If we run out of time, fill remaining splines with the last valid one or a default
            if i < 10 and splines[i+1] is not None:
                splines[i] = splines[i+1]
            else:
                # Fallback: linear spline from K-S to 0
                # This is a very rough heuristic
                x = np.array([0, K, 2*K])
                y = np.array([K, 0, 0])
                splines[i] = UnivariateSpline(x, y, k=1)
            continue
            
        S_i = S_all[:, i]
        V_next = V[i+1, :]
        
        # 1. Sort paths by S_i
        order = np.argsort(S_i)
        S_sorted = S_i[order]
        V_sorted = V_next[order]
        
        # 2. Define bin edges using quantiles to ensure roughly equal sample sizes
        # We want n_bins bins, so n_bins+1 edges.
        # Use linspace(0, 1, n_bins+1) for quantiles.
        # Handle potential duplicate quantiles by adding small jitter or using unique.
        try:
            edges = np.quantile(S_sorted, np.linspace(0, 1, n_bins + 1))
            # Ensure strictly increasing edges for bincount/searchsorted
            # If there are duplicates, nudge them slightly
            for j in range(1, len(edges)):
                if edges[j] <= edges[j-1]:
                    edges[j] = edges[j-1] + 1e-9
        except Exception:
            edges = np.linspace(S_sorted.min(), S_sorted.max(), n_bins + 1)
        
        # 3. Assign each sorted path to a bin
        # np.searchsorted returns the index of the bin edge to the left
        # We want bin index in [0, n_bins-1]
        bin_indices = np.searchsorted(edges, S_sorted, side='right') - 1
        # Clamp to valid range
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # 4. Compute sum of S and sum of V for each bin
        sum_S = np.bincount(bin_indices, weights=S_sorted, minlength=n_bins)
        sum_V = np.bincount(bin_indices, weights=V_sorted, minlength=n_bins)
        counts = np.bincount(bin_indices, minlength=n_bins)
        
        # Avoid division by zero for empty bins (should be rare with quantile edges, but possible)
        valid_bins = counts > 0
        if not np.all(valid_bins):
            # Interpolate missing means from neighbors? Or just skip?
            # For simplicity, if a bin is empty, we can't fit a point there.
            # We will only use valid bins for the spline fit.
            pass
            
        mean_S = np.zeros(n_bins)
        mean_V = np.zeros(n_bins)
        mean_S[valid_bins] = sum_S[valid_bins] / counts[valid_bins]
        mean_V[valid_bins] = sum_V[valid_bins] / counts[valid_bins]
        
        # 5. Fit a spline to the (mean_S, mean_V) points
        # Use UnivariateSpline with smoothing
        # s: smoothing factor. s=0 is interpolation (noisy). Larger s is smoother.
        # A common heuristic is s = n * sigma^2, but we don't know sigma of the error.
        # Let's try s = 0.5 * n_bins or similar. 
        # Actually, UnivariateSpline's s is the sum of squared residuals allowed.
        # Let's try a moderate smoothing.
        try:
            # Use only valid points
            x_fit = mean_S[valid_bins]
            y_fit = mean_V[valid_bins]
            
            # Ensure x is strictly increasing
            # It should be, as mean_S of quantile bins should be increasing
            # But due to noise, it might not be perfectly monotonic.
            # UnivariateSpline requires strictly increasing x.
            # Let's enforce monotonicity on x by sorting (they are already sorted by construction mostly)
            # and removing duplicates.
            
            # Check for duplicates
            unique_mask = np.concatenate(([True], np.diff(x_fit) > 1e-9))
            x_fit = x_fit[unique_mask]
            y_fit = y_fit[unique_mask]
            
            if len(x_fit) < 4:
                raise ValueError("Not enough points for spline")
                
            # Fit spline
            # k=3 (cubic) is standard.
            # s: smoothing. Let's use a small value to allow some smoothing.
            # s = 10 * len(x_fit) ?
            # Let's start with s = 0 (interpolation) and see if it's too noisy.
            # Given the noise in MC, some smoothing is better.
            # Let's try s = 50.
            spline = UnivariateSpline(x_fit, y_fit, k=3, s=50)
            splines[i] = spline
            
            # 6. Evaluate continuation value for all training paths
            C_i = spline(S_i)
            
            # Discounted continuation value
            disc_cont = discount * C_i
            
            # Exercise if K - S_i > disc_cont
            # V[i] is the max of exercise payoff and continuation
            V[i, :] = np.maximum(K - S_i, disc_cont)
            
        except Exception as e:
            # Fallback if spline fitting fails
            # Use linear interpolation on the binned means
            x_fit = mean_S[valid_bins]
            y_fit = mean_V[valid_bins]
            if len(x_fit) > 1:
                # Ensure unique x
                unique_mask = np.concatenate(([True], np.diff(x_fit) > 1e-9))
                x_fit = x_fit[unique_mask]
                y_fit = y_fit[unique_mask]
                spline = UnivariateSpline(x_fit, y_fit, k=1, s=0)
            else:
                spline = UnivariateSpline(np.array([0, K]), np.array([K, 0]), k=1, s=0)
            splines[i] = spline
            C_i = spline(S_i)
            disc_cont = discount * C_i
            V[i, :] = np.maximum(K - S_i, disc_cont)

    # Fill any remaining None splines if the loop broke early
    for i in range(n_dates):
        if splines[i] is None:
            # Use the next available spline or a default
            if i < 10 and splines[i+1] is not None:
                splines[i] = splines[i+1]
            else:
                x = np.array([0, K, 2*K])
                y = np.array([K, 0, 0])
                splines[i] = UnivariateSpline(x, y, k=1)

    class ExercisePolicy:
        def __init__(self, splines_list, n_dates):
            self.splines_list = splines_list
            self.n_dates = n_dates
            self.discount = discount
            self.K = K
            
        def exercise(self, current_state, time_val):
            # time_val is the current time.
            # Exercise dates are at t = 1/12, 2/12, ..., 12/12.
            # Determine the index of the current exercise date.
            # time_val * 12 gives the date number (1 to 12).
            # Index in our arrays (0 to 11) is date_number - 1.
            
            try:
                date_idx = int(round(time_val * 12)) - 1
                if date_idx < 0:
                    date_idx = 0
                if date_idx >= self.n_dates:
                    date_idx = self.n_dates - 1
            except:
                date_idx = 0
                
            S_vals = np.asarray(current_state, dtype=float)
            
            spline = self.splines_list[date_idx]
            
            # Get continuation value for each S
            C = spline(S_vals)
            
            # Discounted continuation
            disc_C = self.discount * C
            
            # Exercise if intrinsic value > continuation
            intrinsic = self.K - S_vals
            
            return intrinsic > disc_C

    policy = ExercisePolicy(splines, n_dates)
    return policy
# EVOLVE-BLOCK-END
