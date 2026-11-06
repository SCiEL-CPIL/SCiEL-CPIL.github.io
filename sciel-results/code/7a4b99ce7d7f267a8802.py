<<<<<<< SEARCH
def _local_letkf_update(local_ens, obs_indices_local, obs_vals, obs_var):
    """
    Perform a proper LETKF update on a local block with multiple observations.
    
    Parameters:
    - local_ens: (n_members, n_local) local ensemble
    - obs_indices_local: array of indices of observed variables within the local block
    - obs_vals: array of observation values
    - obs_var: observation variance (scalar)
    
    Returns:
    - updated local ensemble with preserved spread
    """
    n_members, n_local = local_ens.shape
    n_obs = len(obs_indices_local)
    
    # Compute local ensemble mean and anomalies
    local_mean = np.mean(local_ens, axis=0)
    local_anom = local_ens - local_mean  # (n_members, n_local)
    
    # Extract the observed variables' anomalies
    obs_anom = local_anom[:, obs_indices_local]  # (n_members, n_obs)
    
    # Compute the innovations
    innovations = obs_vals - local_mean[obs_indices_local]  # (n_obs,)
    
    # Compute the ensemble covariance of the observed variables
    # H = obs_anom / (n_members - 1) @ obs_anom.T
    H = np.dot(obs_anom.T, obs_anom) / (n_members - 1)  # (n_obs, n_obs)
    
    # Observation covariance R
    R = obs_var * np.eye(n_obs)
    
    # Compute the Kalman gain matrix K = H @ (H + R)^(-1)
    # K shape: (n_obs, n_obs)
    S = H + R  # (n_obs, n_obs)
    
    # Check if S is positive definite
    try:
        S_inv = np.linalg.inv(S)
        K = np.dot(H, S_inv)  # (n_obs, n_obs)
    except np.linalg.LinAlgError:
        # Fallback: use diagonal approximation
        diag_H = np.diag(H)
        diag_S = diag_H + obs_var
        K = np.diag(diag_H / diag_S)
    
    # Update the mean
    new_mean = local_mean.copy()
    new_mean[obs_indices_local] += np.dot(K, innovations)
    
    # Transform the anomalies to preserve spread
    # For LETKF, we need to find a transform T such that:
    # 1. The new mean is correct
    # 2. The ensemble spread is preserved
    #
    # The standard LETKF approach for multiple observations:
    # Decompose the anomaly matrix using SVD and apply a transform
    
    # For simplicity and stability, we use a sequential update approach
    # where we process each observation one at a time with proper LETKF
    
    new_anom = local_anom.copy()
    
    # Process each observation sequentially
    for j in range(n_obs):
        obs_idx_j = obs_indices_local[j]
        obs_val_j = obs_vals[j]
        innov_j = innovations[j]
        
        # Extract the j-th observed variable's anomalies
        obs_anom_j = local_anom[:, obs_idx_j]
        obs_var_ens_j = np.var(obs_anom_j)
        
        if obs_var_ens_j < 1e-12:
            continue
        
        # Compute the Kalman gain for this single observation
        k_j = obs_var_ens_j / (obs_var_ens_j + obs_var)
        
        # Compute correlations with other local variables
        if obs_var_ens_j > 1e-10:
            cov_with_obs = np.dot(local_anom.T, obs_anom_j) / (n_members - 1)
            corrs_j = cov_with_obs / obs_var_ens_j
        else:
            corrs_j = np.zeros(n_local)
            corrs_j[obs_idx_j] = 1.0
        
        # Update the observed variable's anomalies
        scale_obs = np.sqrt(1.0 - k_j)
        new_anom[:, obs_idx_j] = scale_obs * obs_anom_j
        
        # Update other variables
        for i in range(n_local):
            if i == obs_idx_j:
                continue
            if np.std(obs_anom_j) > 1e-10:
                new_anom[:, i] = new_anom[:, i] + k_j * corrs_j[i] * innov_j * (obs_anom_j / np.std(obs_anom_j))
    
    # Reconstruct the ensemble
    new_ens = new_mean + new_anom
    
    return new_ens
=======
def _local_letkf_update(local_ens, obs_indices_local, obs_vals, obs_var):
    """
    Perform a square-root LETKF update on a local block.
    Uses the transform T = (I - K H) scaled to preserve prior spread.
    """
    n_members, n_local = local_ens.shape
    n_obs = len(obs_indices_local)
    
    local_mean = np.mean(local_ens, axis=0)
    local_anom = local_ens - local_mean  # (n_members, n_local)
    
    # Build H matrix (n_obs, n_local)
    H = np.zeros((n_obs, n_local))
    for i, idx in enumerate(obs_indices_local):
        H[i, idx] = 1.0
        
    # Compute S = H Y Y^T H^T / (n-1) + R
    HY = H @ local_anom  # (n_obs, n_members)
    S = (HY @ HY.T) / (n_members - 1) + obs_var * np.eye(n_obs)
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    # Compute K = P^f H^T S^{-1}
    Pf = (local_anom @ local_anom.T) / (n_members - 1)  # (n_local, n_local)
    K = Pf @ H.T @ S_inv  # (n_local, n_obs)
    
    # Update Mean
    innovations = obs_vals - H @ local_mean
    new_mean = local_mean + K @ innovations
    
    # Update Anomalies using T = (I - K H)
    I = np.eye(n_local)
    T = I - K @ H  # (n_local, n_local)
    new_anom_raw = T @ local_anom  # (n_local, n_members)
    
    # Spread preservation: Scale new_anom_raw to have same trace as local_anom
    prior_spread = np.trace(local_anom @ local_anom.T)
    new_spread = np.trace(new_anom_raw @ new_anom_raw.T)
    
    if new_spread > 1e-12 and prior_spread > 1e-12:
        scale = np.sqrt(prior_spread / new_spread)
        new_anom = scale * new_anom_raw
    else:
        new_anom = new_anom_raw
        
    new_ens = new_mean[:, np.newaxis] + new_anom.T
    return new_ens
>>>>>>> REPLACE

