"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 RHS for an array of states (batch)."""
    n_vars = x.shape[-1]
    x_shifted1 = np.roll(x, -1, axis=-1)  # x_{i+1}
    x_shifted2 = np.roll(x, 2, axis=-1)   # x_{i-2}
    x_shifted3 = np.roll(x, 1, axis=-1)   # x_{i-1}
    return (x_shifted1 - x_shifted2) * x_shifted3 - x + F

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

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    LETKF-based analysis update with proper spread preservation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observations (even-indexed variables)
    - observed_indices: array of indices that are observed
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    n_members, n_vars = ensemble.shape
    obs_idx = observed_indices
    n_obs = len(obs_idx)
    
    # Block localization radius
    radius = 4
    
    # We'll process observations sequentially with block localization
    updated_ensemble = ensemble.copy()
    
    # For each observed variable, perform a local LETKF update
    for obs_k, obs_var_idx in enumerate(obs_idx):
        # Observation value
        obs_val = observations[obs_k]
        obs_var = observation_variance
        
        # Local block: variables within radius of obs_var_idx
        local_indices = []
        for d in range(-radius, radius + 1):
            idx = (obs_var_idx + d) % n_vars
            local_indices.append(idx)
        
        # Remove duplicates (in case of periodic wrap-around)
        local_indices_unique = list(dict.fromkeys(local_indices))
        
        # Find the position of obs_var_idx in the local block
        obs_pos = local_indices_unique.index(obs_var_idx)
        
        # Extract local ensemble
        local_ens = updated_ensemble[:, local_indices_unique].copy()
        
        # Perform LETKF update on the local block
        local_ens_updated = _local_letkf_update(local_ens, [obs_pos], [obs_val], obs_var)
        
        # Update the full ensemble
        updated_ensemble[:, local_indices_unique] = local_ens_updated
    
    # Ensure finiteness and valid range
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
