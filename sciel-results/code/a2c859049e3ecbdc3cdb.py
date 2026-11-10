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

def _periodic_distance(i, j, n):
    """Compute periodic distance between indices i and j on a ring of size n."""
    d = abs(i - j)
    return min(d, n - d)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Transform Kalman Filter (ETKF) update for Lorenz-96 assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations for observed_indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    updated ensemble with same shape (M, N)
    """
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Start with a copy of the ensemble
    result = ensemble.copy()
    
    # Adaptive Inflation with better calibration
    if M > 1:
        # Compute spread at observed locations
        spread = np.std(ensemble[:, observed_indices], axis=0)
        obs_err = np.sqrt(observation_variance)
        # Use median ratio for robustness
        ratios = spread / obs_err if obs_err > 0 else np.ones_like(spread)
        median_ratio = np.median(ratios)
        
        # Target spread-to-error ratio around 1.0-1.5
        # If ratio is low, inflate more; if high, inflate less
        if median_ratio < 1.0:
            # Need more spread
            inflation = 1.0 + 0.1 * (1.0 - median_ratio)
            inflation = np.clip(inflation, 1.01, 1.15)
        elif median_ratio < 1.5:
            # Good spread, minimal inflation
            inflation = 1.01
        else:
            # Too much spread, deflate slightly
            inflation = 1.0 - 0.05 * (median_ratio - 1.5)
            inflation = np.clip(inflation, 0.95, 1.01)
        
        # Apply multiplicative inflation around the mean
        mean = np.mean(result, axis=0)
        result = mean + inflation * (result - mean)
    else:
        inflation = 1.01

    # Local ETKF Update with improved localization
    # Use a wider window for better localization
    window_half = 3
    window_size = 2 * window_half + 1
    
    # Pre-compute observation index lookup for faster access
    obs_dict = {idx: val for idx, val in zip(observed_indices, observations)}
    
    # Iterate over each state variable i
    for i in range(N):
        # Determine local window indices with periodic boundary
        local_indices = [(i + j) % N for j in range(-window_half, window_half + 1)]
        
        # Identify which indices in the window are observed
        local_obs_mask = np.zeros(window_size, dtype=bool)
        local_obs_vals = np.zeros(window_size)
        
        for l_idx, g_idx in enumerate(local_indices):
            if g_idx in obs_dict:
                local_obs_mask[l_idx] = True
                local_obs_vals[l_idx] = obs_dict[g_idx]
        
        n_obs_local = np.sum(local_obs_mask)
        
        # If no observations in this local window, keep the state as is
        if n_obs_local == 0:
            continue
        
        # Extract local ensemble states: shape (M, window_size)
        local_ens = result[:, local_indices]
        
        # Extract local observations: shape (n_obs_local,)
        y_local = local_obs_vals[local_obs_mask]
        
        # Compute local ensemble mean and anomalies
        local_mean = np.mean(local_ens, axis=0)
        local_anoms = local_ens - local_mean
        
        # Compute local ensemble covariance (M-1 normalization)
        C_local = (local_anoms.T @ local_anoms) / (M - 1)
        
        # Apply localization to covariance
        # Build localization matrix based on periodic distances
        loc_matrix = np.zeros((window_size, window_size))
        for a in range(window_size):
            for b in range(window_size):
                dist = _periodic_distance(local_indices[a], local_indices[b], N)
                loc_matrix[a, b] = _gaspari_cohn(dist, c=4.0)
        
        # Localized covariance: C_loc = L * C * L (element-wise)
        C_localized = C_local * loc_matrix
        
        # Observation operator H_local: selects observed components
        H_local = np.zeros((n_obs_local, window_size))
        for k, l_idx in enumerate(np.where(local_obs_mask)[0]):
            H_local[k, l_idx] = 1.0
        
        # R = observation_variance * I
        R_local = observation_variance * np.eye(n_obs_local)
        
        # Innovation covariance S = H C_loc H^T + R
        HC = H_local @ C_localized
        S = HC @ H_local.T + R_local
        
        # Compute Gain K = C_loc H^T S^-1
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        K_local = C_localized @ H_local.T @ S_inv
        
        # Innovation y - H x_mean
        innovation = y_local - H_local @ local_mean
        
        # Deterministic update for each member in the local window
        for m in range(M):
            Hx_m = H_local @ local_ens[m, :]
            local_ens_new_m = local_ens[m, :] + K_local @ (y_local - Hx_m)
            result[m, local_indices] = local_ens_new_m
        
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
