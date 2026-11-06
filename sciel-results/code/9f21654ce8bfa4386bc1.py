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
    H = np.dot(obs_anom.T, obs_anom) / (n_members - 1)  # (n_obs, n_obs)
    
    # Observation covariance R
    R = obs_var * np.eye(n_obs)
    
    # Compute the Kalman gain matrix K = H @ (H + R)^(-1)
    S = H + R  # (n_obs, n_obs)
    
    try:
        S_inv = np.linalg.inv(S)
        K = np.dot(H, S_inv)  # (n_obs, n_obs)
    except np.linalg.LinAlgError:
        diag_H = np.diag(H)
        diag_S = diag_H + obs_var
        K = np.diag(diag_H / diag_S)
    
    # Update the mean
    new_mean = local_mean.copy()
    new_mean[obs_indices_local] += np.dot(K, innovations)
    
    # Compute the full local forecast covariance P_f
    P_f = np.dot(local_anom.T, local_anom) / (n_members - 1)  # (n_local, n_local)
    
    # Compute the analysis covariance P_a
    # P_a = (I - K H) P_f (I - K H)^T + K R K^T
    # H is the observation operator for the local block.
    # In this context, obs_indices_local maps global local indices to observation slots.
    # We need to construct H_local which is (n_obs, n_local)
    H_local = np.zeros((n_obs, n_local))
    for i, idx in enumerate(obs_indices_local):
        H_local[i, idx] = 1.0
        
    # K is (n_obs, n_obs). We need K_H which is (n_local, n_local)?
    # The update is x_a = x_f + K (y - H x_f).
    # The gain matrix for the state is G = K H.
    # G is (n_local, n_obs) * (n_obs, n_local) ? No.
    # K is (n_obs, n_obs). H is (n_obs, n_local).
    # The state gain is G = K H? No, dimensions: (n_obs, n_obs) * (n_obs, n_local) is invalid.
    # The Kalman gain for the state is K_state = P_f H^T (H P_f H^T + R)^-1.
    # K_state is (n_local, n_obs).
    # Here, K computed above is (n_obs, n_obs). This is the gain for the observations?
    # No, standard Kalman: K = P H^T (H P H^T + R)^-1.
    # My K is H (H + R)^-1? No, I computed K = H S^-1 where S = H P_obs H^T + R?
    # In my code: H is the covariance of the observed variables.
    # H = P_obs.
    # S = P_obs + R.
    # K = P_obs (P_obs + R)^-1.
    # This K is the gain for the observed variables in the innovation space?
    # The update for the observed variables is x_obs_a = x_obs_f + K (y - x_obs_f).
    # This is correct for the observed variables.
    # For the full state, we need the cross-covariance.
    
    # Let's compute the full state gain properly.
    # P_f is (n_local, n_local).
    # H_local is (n_obs, n_local).
    # P_f H_local^T is (n_local, n_obs).
    # S = H_local P_f H_local^T + R.
    # We already have S as H + R, where H = P_obs = H_local P_f H_local^T.
    # So S is correct.
    # K_state = P_f H_local^T S^-1.
    
    try:
        K_state = np.dot(P_f, np.dot(H_local.T, S_inv))
    except:
        K_state = np.zeros((n_local, n_obs))
        
    # Update the full mean using K_state
    new_mean_full = local_mean + np.dot(K_state, innovations)
    
    # Compute P_a
    # P_a = (I - K_state H_local) P_f (I - K_state H_local)^T + K_state R K_state^T
    I_local = np.eye(n_local)
    M = I_local - np.dot(K_state, H_local)
    P_a = np.dot(np.dot(M, P_f), M.T) + np.dot(np.dot(K_state, R), K_state.T)
    
    # Ensure P_a is symmetric and positive semi-definite
    P_a = (P_a + P_a.T) / 2.0
    try:
        L_a = np.linalg.cholesky(P_a + 1e-10 * np.eye(n_local))
    except:
        L_a = np.linalg.cholesky(P_f + 1e-10 * np.eye(n_local))
        
    try:
        L_f = np.linalg.cholesky(P_f + 1e-10 * np.eye(n_local))
    except:
        L_f = np.eye(n_local)
        
    # Compute transformation T such that T P_f T^T = P_a
    # T = L_m^-1 L_f^-1 where L_m = chol( (L_f^-1 L_a) (L_f^-1 L_a)^T )
    try:
        M_trans = np.dot(np.linalg.inv(L_f), L_a)
        L_m = np.linalg.cholesky(np.dot(M_trans, M_trans.T) + 1e-10 * np.eye(n_local))
        T = np.dot(np.linalg.inv(L_m), np.linalg.inv(L_f))
    except:
        T = np.eye(n_local)
        
    # Apply transformation to anomalies
    # new_anom = local_anom @ T.T ?
    # x_a = x_f + T (x_f - x_f_mean)?
    # If we define new_anom = T @ local_anom.T, then P_a_new = T P_f T^T.
    # local_anom is (n_members, n_local).
    # local_anom.T is (n_local, n_members).
    # T is (n_local, n_local).
    # new_anom.T = T @ local_anom.T
    # new_anom = local_anom @ T.T
    
    new_anom = np.dot(local_anom, T.T)
    
    # Reconstruct the ensemble
    new_ens = new_mean_full + new_anom
    
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
