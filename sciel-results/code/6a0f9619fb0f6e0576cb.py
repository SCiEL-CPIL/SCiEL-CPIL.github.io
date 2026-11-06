"""Bermudan put exercise policy: Parametric smooth boundary via polynomial fit to backward induction thresholds."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy by:
    1. Running backward induction on a fine grid to estimate optimal boundaries.
    2. Fitting a smooth monotonic polynomial to the extracted boundaries.
    3. Using the polynomial to make exercise decisions.
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
    
    # Use a finer grid for internal backward induction to get accurate boundaries
    # 256 bins should be sufficient for smooth boundary extraction
    n_bins = 256
    
    # Compute bin edges from training data using quantiles across all dates
    all_prices = training_paths.flatten()
    positive_prices = all_prices[all_prices > 0]
    if len(positive_prices) < n_bins:
        price_min = max(0.01, all_prices.min() * 0.9 if all_prices.size > 0 else 0.01)
        price_max = all_prices.max() * 1.1 if all_prices.size > 0 else 200.0
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    else:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(positive_prices, quantiles)
        # Ensure strictly increasing
        for i in range(1, len(bin_edges)):
            if bin_edges[i] <= bin_edges[i-1]:
                bin_edges[i] = bin_edges[i-1] + 1e-6
        # Extend edges slightly to cover full range
        bin_edges[0] = max(0.01, bin_edges[0] * 0.95)
        bin_edges[-1] = bin_edges[-1] * 1.05

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_edges_expanded = np.concatenate([[-np.inf], bin_edges[1:-1], [np.inf]])

    # Backward induction
    # V_current holds V_{i+1}(S_{i+1}) for each path at the current iteration
    # Start at last exercise date: V = max(K - S, 0)
    V_current = np.maximum(K - training_paths[:, -1], 0.0)
    
    # Store the estimated boundary S*_i for each date i (0 to n_dates-2)
    # S*_i is the spot price below which we exercise at date i
    boundaries = [None] * n_dates
    boundaries[-1] = K  # At maturity, exercise if S < K

    # Process dates from n_dates-2 down to 0
    for i in range(n_dates - 2, -1, -1):
        if time.time() > deadline - 3:
            # Break early, use default policy for remaining dates
            # Fill remaining boundaries with K (conservative) or linear interp
            for j in range(i, -1, -1):
                if boundaries[j] is None:
                    boundaries[j] = K
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

        # For bins with no data, interpolate
        if np.any(bin_counts == 0):
            valid_mask = bin_counts > 0
            if np.sum(valid_mask) >= 2:
                valid_indices = np.where(valid_mask)[0]
                valid_values = bin_means[valid_mask]
                # Interpolate for empty bins
                bin_means = np.interp(np.arange(n_bins), valid_indices, valid_values)
            else:
                overall_mean = np.mean(V_current)
                bin_means = np.where(bin_counts > 0, bin_means, overall_mean)

        # Apply light smoothing to reduce noise in continuation values
        if n_bins >= 3:
            smoothed = np.zeros(n_bins)
            smoothed[0] = bin_means[0]
            smoothed[-1] = bin_means[-1]
            for b in range(1, n_bins - 1):
                smoothed[b] = (bin_means[b-1] + 2*bin_means[b] + bin_means[b+1]) / 4.0
            bin_means = smoothed

        # Get continuation value for each path
        cont_value = bin_means[bin_indices]
        continue_value = disc * cont_value

        # Exercise payoff
        exercise_payoff = np.maximum(K - S_i, 0.0)

        # Optimal value
        V_i = np.maximum(exercise_payoff, continue_value)

        # Update V_current for next iteration
        V_current = V_i
        
        # Extract boundary S*_i
        # We want S such that K - S = disc * V_{i+1}(S)
        # Let's evaluate this difference at the bin centers
        S_grid = bin_centers
        payoff_grid = np.maximum(K - S_grid, 0.0)
        cont_grid = disc * bin_means
        diff = payoff_grid - cont_grid
        
        # Find the index where diff changes sign from positive to negative
        # Since payoff decreases with S and cont generally increases (or varies), 
        # diff is likely positive for low S and negative for high S.
        # We look for the first index where diff < 0.
        neg_indices = np.where(diff < 0)[0]
        
        if len(neg_indices) > 0:
            idx = neg_indices[0]
            if idx == 0:
                boundaries[i] = bin_centers[0]
            else:
                # Linear interpolation between idx-1 and idx
                S1 = bin_centers[idx-1]
                S2 = bin_centers[idx]
                d1 = diff[idx-1]
                d2 = diff[idx]
                if d1 != d2:
                    boundaries[i] = S1 + (S2 - S1) * (-d1) / (d2 - d1)
                else:
                    boundaries[i] = S1
        else:
            # If diff is always positive, exercise everywhere? Unlikely for puts.
            # If diff is always negative, never exercise? Unlikely.
            # Check if diff is all positive -> boundary is max price
            if np.all(diff > 0):
                boundaries[i] = bin_centers[-1]
            else:
                # If all negative, boundary is 0
                boundaries[i] = 0.0

    # Ensure boundaries are filled
    for j in range(n_dates):
        if boundaries[j] is None:
            boundaries[j] = K

    # Fit a smooth monotonic polynomial to the boundaries
    # Time to maturity tau = T - i * dt
    # We fit log(boundary) or just boundary as a function of tau.
    # Boundary is typically between 0 and K.
    # Let's fit a polynomial in tau: B(tau) = c0 + c1*tau + c2*tau^2
    # We want B(tau) to be non-decreasing in tau (as time to maturity increases, boundary increases? 
    # Wait. 
    # t=0 (start), tau=T. Boundary is low (e.g., 50).
    # t=T (end), tau=0. Boundary is K (100).
    # So as tau decreases, Boundary increases.
    # So B(tau) is non-increasing in tau? 
    # Let's check: 
    # Put option: Exercise if S < B(t). 
    # Early (t=0, tau=T), we only exercise if S is very low. B(0) is small.
    # Late (t=T, tau=0), we exercise if S < K. B(T) is K.
    # So B(t) increases with t. 
    # Since tau = T - t, B(tau) decreases with tau.
    # Let's fit B(tau) = c0 - c1*tau + c2*tau^2 with c1 > 0? 
    # Or just fit B(t) = c0 + c1*t + c2*t^2 with c1 > 0.
    
    t_values = np.linspace(0, T, n_dates)
    b_values = np.array(boundaries)
    
    # Fit a quadratic polynomial B(t) = c0 + c1*t + c2*t^2
    # Using least squares
    A = np.vstack([np.ones(n_dates), t_values, t_values**2]).T
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A, b_values, rcond=None)
        c0, c1, c2 = coeffs
    except:
        # Fallback to linear
        A_lin = np.vstack([np.ones(n_dates), t_values]).T
        coeffs_lin, _, _, _ = np.linalg.lstsq(A_lin, b_values, rcond=None)
        c0, c1 = coeffs_lin
        c2 = 0.0

    # Optional: Enforce monotonicity? 
    # The derivative is c1 + 2*c2*t. 
    # If the fit is noisy, it might not be monotonic. 
    # However, the backward induction boundaries should be reasonably monotonic.
    # We can project the coefficients to ensure monotonicity if needed, 
    # but for now, the smooth fit should be better than the raw noisy boundaries.

    class ExercisePolicy:
        def __init__(self, c0, c1, c2, K, n_dates, T):
            self.c0 = c0
            self.c1 = c1
            self.c2 = c2
            self.K = K
            self.n_dates = n_dates
            self.T = T

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
            
            # Calculate the time value t from the index
            # time is the index of the exercise date (0 to n_dates-1)
            # t = time * (T / n_dates)
            t = time * (self.T / self.n_dates)
            
            # Calculate the boundary B(t)
            B_t = self.c0 + self.c1 * t + self.c2 * t * t
            
            # Ensure boundary is within [0, K]
            B_t = np.clip(B_t, 0.0, self.K)
            
            # Exercise if S < B_t
            return current_state < B_t

    return ExercisePolicy(c0, c1, c2, K, n_dates, T)
# EVOLVE-BLOCK-END
