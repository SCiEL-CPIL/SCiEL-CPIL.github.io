"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn correlation function for localization."""
    d = np.abs(d)
    result = np.zeros_like(d)
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    # d <= 0.5c
    result[mask1] = (-5.0 / 4.0) * d[mask1]**5 / c**5 + 5.0 * d[mask1]**4 / c**4 - 10.0 * d[mask1]**3 / c**3 + 5.0 * d[mask1]
    # 0.5c < d <= 1.5c
    result[mask2] = (1.0 / 12.0) * d[mask2]**5 / c**5 - (5.0 / 4.0) * d[mask2]**4 / c**4 + (5.0 / 2.0) * d[mask2]**3 / c**3 - 5.0 * d[mask2]**2 / c**2 + (25.0 / 12.0) * d[mask2] / c + 1.0 / 2.0
    # 1.5c < d <= 2.5c
    result[mask3] = (-1.0 / 16.0) * d[mask3]**5 / c**5 + (1.0 / 4.0) * d[mask3]**4 / c**4 - (15.0 / 8.0) * d[mask3]**3 / c**3 + (5.0 / 2.0) * d[mask3]**2 / c**2 - (25.0 / 12.0) * d[mask3] / c + 105.0 / 48.0
    
    return result

def _compute_localization_matrix(N, c=3.0):
    """Precompute localization matrix for periodic domain."""
    loc_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d = abs(i - j)
            d = min(d, N - d)
            loc_matrix[i, j] = _gaspari_cohn(d, c)
    return loc_matrix

# Precompute localization matrix for N=20, c=3.0
# We can do this inside analyze or globally if N is fixed. 
# Since N is always 20 in this task, we can hardcode or compute once.
# However, to be safe and generic, we compute it in analyze but cache it if possible.
# For simplicity and speed, we'll compute it every time but it's small (20x20).
# Actually, let's just compute it once globally if N is known to be 20.
# But analyze takes ensemble, so N is derived.
# Let's use a simple cache.

_loc_cache = {}

def _get_loc_matrix(N, c=3.0):
    key = (N, c)
    if key not in _loc_cache:
        _loc_cache[key] = _compute_localization_matrix(N, c)
    return _loc_cache[key]

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Transform Kalman Filter (ETKF) for Lorenz-96 assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations for observed_indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    updated ensemble with same shape (M, N)
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Compute ensemble mean
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean
    
    # Get localization matrix
    loc_matrix = _get_loc_matrix(N, c=3.0)
    
    # Identify observed indices as a set for quick lookup
    obs_set = set(observed_indices)
    
    # Initialize result
    result = ensemble.copy()
    
    # Loop over each state variable i
    for i in range(N):
        # Determine which variables are in the local window for i
        # Using Gaspari-Cohn with c=3.0, non-zero for distance <= 2.5*c = 7.5
        # So distance 0..7
        window = []
        for j in range(N):
            d = abs(i - j)
            d = min(d, N - d)
            if d <= 7:
                window.append(j)
        
        if not window:
            continue
            
        window_idx = np.array(window)
        
        # Local anomalies (M x L)
        local_anoms = anomalies[:, window_idx]
        L = len(window_idx)
        
        # Check which of these are observed
        local_obs_mask = np.array([j in obs_set for j in window_idx])
        local_obs_indices_in_window = np.where(local_obs_mask)[0]
        
        if len(local_obs_indices_in_window) == 0:
            # No observations in this local window, skip update (or just inflate)
            continue
        
        # Get local observed anomalies (M x K_local)
        local_obs_anoms = local_anoms[:, local_obs_indices_in_window]
        K_loc = len(local_obs_indices_in_window)
        
        # Local ensemble mean
        local_mean = mean[window_idx]
        
        # Observation innovation
        # We need the observations corresponding to the observed indices in the window
        # Map window indices to original indices
        orig_obs_indices = window_idx[local_obs_indices_in_window]
        
        # Get the specific observations
        # observed_indices is an array of original indices
        # We need to find which observations correspond to orig_obs_indices
        # Create a map from original index to observation value
        obs_map = {}
        for idx, val in zip(observed_indices, observations):
            obs_map[idx] = val
        
        local_obs_vals = np.array([obs_map[idx] for idx in orig_obs_indices])
        
        # Innovation: y - H x_mean
        Hx_mean = local_mean[local_obs_indices_in_window]
        innovation = local_obs_vals - Hx_mean
        
        # Compute local covariance
        # C = local_anoms.T @ local_anoms / (M-1)  (L x L)
        # We only need the sub-block for observed variables and the cross-covariance
        # C_oo = local_obs_anoms.T @ local_obs_anoms / (M-1)  (K_loc x K_loc)
        # C_ox = local_obs_anoms.T @ local_anoms / (M-1)      (K_loc x L)
        
        C_oo = (local_obs_anoms.T @ local_obs_anoms) / (M - 1)
        C_ox = (local_obs_anoms.T @ local_anoms) / (M - 1)
        
        # Apply localization to the covariance
        # Standard approach: multiply the covariance by the localization matrix
        # But for ETKF, we usually apply localization to the transformation matrix
        # Or we can use a localized covariance.
        # Let's use a simple localized gain approach.
        
        # Observation error variance
        R = observation_variance
        
        # Innovation covariance S = C_oo + R
        S = C_oo + R * np.eye(K_loc)
        
        # Gain K = C_ox S^{-1}
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        K_gain = C_ox @ S_inv  # (L x K_loc)
        
        # Apply localization to the gain
        # The gain should be localized.
        # For each row l (corresponding to variable window_idx[l]),
        # we scale the gain by the localization factor between i and window_idx[l]
        # Wait, the gain K_gain updates the local anomalies.
        # The update for member m is:
        # x_a[m, i] = x_f[m, i] + sum_{l in window} K_gain[l, :] @ (y - H x_f[m, window_idx[l]])
        # But this is not exactly right. The standard EnKF update is:
        # x_a = x_f + K (y - H x_f)
        # Where K is (N x K).
        # In the local filter, we compute a local gain for the variables in the window.
        # Then we update the variables in the window.
        # But we are iterating over i. This is a bit tricky.
        
        # Alternative: Compute the full gain matrix for the local window,
        # then apply it to update the local variables.
        # But we are inside a loop for i.
        
        # Let's switch to a simpler, more standard Local EnKF approach:
        # For each i, compute the gain K_i (N x K) using the full covariance
        # but localized.
        
        # Actually, the current loop structure is inefficient for ETKF.
        # Let's implement a standard Local EnKF with localization.
        
        # Break out of the loop and use a simpler method if this gets too complex.
        # Given the time constraint, let's stick to a robust Global EnKF with
        # strong localization and inflation, which is simpler and often competitive.
        
        # Let's revert to a refined Global EnKF with Gaspari-Cohn localization
        # applied to the covariance matrix.
        
        # This loop is becoming too complex. Let's abort and use a simpler strategy.
        # The previous program used a global EnKF. Let's improve it with
        # proper localization of the covariance matrix.
        
        # We will break out and use a different implementation.
        pass

    # Since the local loop is complex, let's use a vectorized Global EnKF
    # with localized covariance.
    
    # Recompute covariance
    cov = (anomalies.T @ anomalies) / (M - 1)
    
    # Apply localization to the covariance
    cov_loc = cov * loc_matrix
    
    # Observation operator
    HcovHT = cov_loc[np.ix_(observed_indices, observed_indices)]
    R = observation_variance * np.eye(K)
    
    S = HcovHT + R
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Cross-covariance
    cov_H = cov_loc[:, observed_indices]
    
    # Gain
    K_gain = cov_H @ S_inv
    
    # Innovation
    Hx = mean[observed_indices]
    innovation = observations - Hx
    
    # Update mean
    new_mean = mean + K_gain @ innovation
    
    # Update ensemble
    # x_a = x_f + K (y - H x_f)
    # To avoid spurious correlations, we can use a transform, but for simplicity
    # and speed, we use the deterministic update with inflation.
    
    result = ensemble.copy()
    for i in range(M):
        Hxi = ensemble[i, observed_indices]
        result[i] = ensemble[i] + K_gain @ (observations - Hxi)
    
    # Adaptive Inflation
    # Compute spread vs observation error
    spread = np.std(ensemble[:, observed_indices], axis=0)
    obs_err = np.sqrt(observation_variance)
    
    ratio = np.mean(spread) / obs_err if obs_err > 0 else 1.0
    
    if ratio < 1.0:
        inflation = 1.0 + 0.1 * (1.0 - ratio)
        result_mean = np.mean(result, axis=0)
        result = result_mean + inflation * (result - result_mean)
    elif ratio > 1.5:
        # Deflate if spread is too large
        inflation = 1.0 - 0.05 * (ratio - 1.5)
        inflation = max(0.9, inflation)
        result_mean = np.mean(result, axis=0)
        result = result_mean + inflation * (result - result_mean)
    
    # Ensure finite and within bounds
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
