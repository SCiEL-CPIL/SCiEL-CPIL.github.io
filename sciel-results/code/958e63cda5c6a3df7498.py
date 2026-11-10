"""Local ESRF analysis with Gaspari-Cohn localization and adaptive inflation for Lorenz-96."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gaspari_cohn_weights(distances, c):
    """
    Compute Gaspari-Cohn localization weights.
    
    Parameters:
    distances: array of distances from center
    c: correlation length (half-width of support)
    
    Returns:
    array of weights
    """
    r = np.asarray(distances, dtype=float) / c
    result = np.zeros_like(r)
    
    # r <= 0.5
    m1 = r <= 0.5
    r1 = r[m1]
    result[m1] = (-5.0/4.0)*r1**5 + 5.0*r1**4 - 10.0*r1**3 + 5.0*r1
    
    # 0.5 < r <= 1.5
    m2 = (r > 0.5) & (r <= 1.5)
    r2 = r[m2]
    result[m2] = (1.0/12.0)*r2**5 - (5.0/4.0)*r2**4 + (5.0/2.0)*r2**3 - 5.0*r2**2 + (25.0/12.0)*r2 + 0.5
    
    # 1.5 < r <= 2.5
    m3 = (r > 1.5) & (r <= 2.5)
    r3 = r[m3]
    result[m3] = (-1.0/16.0)*r3**5 + (1.0/4.0)*r3**4 - (15.0/8.0)*r3**3 + (5.0/2.0)*r3**2 - (25.0/12.0)*r3 + 105.0/48.0
    
    # Ensure center weight is exactly 1
    if len(result) > 0:
        center = len(result) // 2
        result[center] = 1.0
    
    return result


def _local_esrf_update(local_ens, local_obs_mask, y_local, obs_var, window_half, N):
    """
    Perform a local Ensemble Square Root Filter update with Gaspari-Cohn localization.
    
    Parameters:
    local_ens: (M, L) array of local ensemble states
    local_obs_mask: (L,) boolean array indicating which local variables are observed
    y_local: (n_obs,) array of local observations
    obs_var: scalar observation variance
    window_half: half-window size (for computing distances)
    N: total number of state variables (for periodic distance)
    
    Returns:
    updated local ensemble (M, L)
    """
    M, L = local_ens.shape
    n_obs = len(y_local)
    
    if n_obs == 0:
        return local_ens.copy()
    
    # Ensemble mean
    x_mean = np.mean(local_ens, axis=0)
    
    # Anomalies
    A = local_ens - x_mean  # (M, L)
    
    # Compute Gaspari-Cohn weights for localization
    # The local window is centered at some index i, positions are -window_half to +window_half
    distances = np.array([min(abs(j), N - abs(j)) for j in range(-window_half, window_half + 1)], dtype=float)
    corr_len = float(window_half + 1)  # correlation length slightly larger than window
    weights = _gaspari_cohn_weights(distances, corr_len)
    
    # Apply localization weights to anomalies
    A_loc = A * weights[np.newaxis, :]  # (M, L)
    
    # Local observation operator H: (n_obs, L)
    H = np.zeros((n_obs, L))
    obs_idx_in_local = np.where(local_obs_mask)[0]
    for k, idx in enumerate(obs_idx_in_local):
        H[k, idx] = 1.0
    
    # Innovation
    innovation = y_local - H @ x_mean  # (n_obs,)
    
    # Local covariance with localization C = A_loc^T A_loc / (M-1)
    C = (A_loc.T @ A_loc) / (M - 1)  # (L, L)
    
    # Innovation covariance S = H C H^T + R
    HC = H @ C  # (n_obs, L)
    S = HC @ H.T + obs_var * np.eye(n_obs)  # (n_obs, n_obs)
    
    # Compute gain K = C H^T S^{-1}
    HC_T = H @ C.T  # (n_obs, L)
    try:
        Z_T = np.linalg.solve(S, HC_T)
        K = Z_T.T  # (L, n_obs)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = C @ H.T @ S_inv
    
    # Update each member
    updated_ens = np.zeros_like(local_ens)
    for m in range(M):
        Hx = H @ local_ens[m, :]
        updated_ens[m, :] = local_ens[m, :] + K @ (y_local - Hx)
    
    return updated_ens


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Square Root Filter (ESRF) update for Lorenz-96 assimilation
    with Gaspari-Cohn localization and adaptive inflation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations for observed_indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    updated ensemble with same shape (M, N)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    if K == 0:
        return ensemble.copy()
    
    # Start with a copy of the ensemble
    result = ensemble.copy()
    
    # Adaptive Inflation
    if M > 1:
        spread = np.std(ensemble[:, observed_indices], axis=0)
        obs_err = np.sqrt(observation_variance)
        ratio = np.mean(spread) / obs_err if obs_err > 0 else 1.0
        
        # If spread is too small, inflate more aggressively
        if ratio < 1.0:
            inflation = 1.0 + 0.15 * (1.0 - ratio)
            inflation = np.clip(inflation, 1.01, 1.15)
        else:
            inflation = 1.0 + 0.05 * np.exp(-(ratio - 1.0))
            inflation = np.clip(inflation, 1.01, 1.05)
        
        # Apply multiplicative inflation around the mean
        mean = np.mean(result, axis=0)
        result = mean + inflation * (result - mean)
    else:
        inflation = 1.01

    # Local ESRF Update
    # Window size: include the target variable and its neighbors
    window_half = 3
    window_size = 2 * window_half + 1
    
    # Map global indices to observed values for quick lookup
    obs_lookup = {}
    for i, idx in enumerate(observed_indices):
        obs_lookup[idx] = observations[i]
    
    # Iterate over each state variable i
    for i in range(N):
        if time.time() > deadline:
            break
        
        # Determine local window indices
        local_indices = [(i + j) % N for j in range(-window_half, window_half + 1)]
        
        # Identify which indices in the window are observed
        local_obs_mask = np.zeros(window_size, dtype=bool)
        local_obs_vals = np.zeros(window_size)
        
        for l_idx, g_idx in enumerate(local_indices):
            if g_idx in obs_lookup:
                local_obs_mask[l_idx] = True
                local_obs_vals[l_idx] = obs_lookup[g_idx]
        
        n_obs_local = np.sum(local_obs_mask)
        
        # If no observations in this local window, keep the state as is
        if n_obs_local == 0:
            continue
        
        # Extract local ensemble states: shape (M, window_size)
        local_ens = result[:, local_indices]
        
        # Extract local observations: shape (n_obs_local,)
        y_local = local_obs_vals[local_obs_mask]
        
        # Perform local ESRF update with Gaspari-Cohn localization
        updated_local_ens = _local_esrf_update(local_ens, local_obs_mask, y_local, observation_variance, window_half, N)
        
        # Update result
        result[:, local_indices] = updated_local_ens
    
    # Final safety checks
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result

# EVOLVE-BLOCK-END
