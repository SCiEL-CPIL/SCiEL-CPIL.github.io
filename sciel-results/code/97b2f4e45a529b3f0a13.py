"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Smoother (LEST) with adaptive multiplicative inflation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,) or (16, 10)
        Noisy observations of the observed indices.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Normalize observations to a single vector of length n_obs
    if observations.ndim == 1:
        y = observations
    else:
        y = np.mean(observations, axis=0)
        
    # Initialize updated ensemble
    updated_ensemble = ensemble.copy()
    
    # Local window size: use full state for better global consistency
    # With 20 variables and 10 observations, using a larger window helps
    local_window = 11
    half_window = local_window // 2
    
    # Loop over each observation
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        obs_val = y[j]
        
        # Determine local indices (periodic)
        local_indices = []
        for k in range(-half_window, half_window + 1):
            local_indices.append((obs_idx + k) % n_vars)
            
        # Extract local ensemble states: shape (n_members, local_window)
        x_local = updated_ensemble[:, local_indices]
        
        # Compute local mean and covariance
        x_local_mean = np.mean(x_local, axis=0)
        x_local_anom = x_local - x_local_mean
        P_local = (x_local_anom.T @ x_local_anom) / (n_members - 1)
        
        # Add small regularization to prevent ill-conditioning
        reg = 1e-6
        P_local += reg * np.eye(local_window)
        
        # Observation operator H_local
        H_local = np.zeros(local_window)
        H_local[half_window] = 1.0
        
        # Compute Kalman Gain
        HP = P_local @ H_local
        S = H_local @ HP + observation_variance
        
        if S < 1e-12:
            K_local = np.zeros(local_window)
        else:
            K_local = HP / S
            
        # Innovation for each member
        Hx_local = x_local[:, half_window]
        innovations = obs_val - Hx_local
        
        # Update local ensemble members
        update_matrix = np.outer(innovations, K_local)
        x_local_updated = x_local + update_matrix
        
        updated_ensemble[:, local_indices] = x_local_updated
    
    # Adaptive Multiplicative Inflation
    # Compute the expected innovation variance from ensemble spread and obs noise
    # Compare with actual innovation variance to adjust inflation
    mean_ens = np.mean(updated_ensemble, axis=0)
    
    # Compute innovation statistics for observed variables
    innovation_squares = []
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        ens_spread = np.std(updated_ensemble[:, obs_idx]) ** 2
        obs_noise_var = observation_variance
        expected_innov_var = ens_spread + obs_noise_var
        actual_innov = y[j] - updated_ensemble[:, obs_idx]
        actual_innov_var = np.mean(actual_innov ** 2)
        innovation_squares.append(actual_innov_var / max(expected_innov_var, 1e-10))
    
    # Use mean ratio to determine inflation
    mean_ratio = np.mean(innovation_squares)
    
    # Target inflation: if ratio > 1, ensemble is under-dispersed -> inflate more
    # if ratio < 1, ensemble is over-dispersed -> deflate
    # Use a smooth mapping
    base_inflation = 1.02
    adaptive_factor = 1.0 + 0.05 * (mean_ratio - 1.0)
    inflation_factor = base_inflation * adaptive_factor
    
    # Clip inflation to reasonable range
    inflation_factor = np.clip(inflation_factor, 0.98, 1.08)
    
    # Apply inflation
    updated_ensemble = inflation_factor * (updated_ensemble - mean_ens) + mean_ens
    
    # Ensure finite values and clip
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
