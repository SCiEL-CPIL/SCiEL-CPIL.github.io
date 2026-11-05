"""Lorenz-96 assimilation using Local Ensemble Kalman Filter with localization and inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _localization_matrix(obs_idx, n_vars):
    """
    Compute localization weights for the cross-covariance.
    Uses a simple exponential decay based on index distance.
    """
    n_obs = len(obs_idx)
    # We need a matrix of shape (n_vars, n_obs) where L[i, j] is weight for variable i given observation j
    # Distance is circular distance on the ring of size n_vars
    weights = np.zeros((n_vars, n_obs))
    for i in range(n_vars):
        for j in range(n_obs):
            dist = min(abs(i - obs_idx[j]), n_vars - abs(i - obs_idx[j]))
            # Decay length scale: typically 2-5 variables for L96
            decay_len = 3.0
            weights[i, j] = np.exp(-dist / decay_len)
    return weights

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an ensemble Kalman filter update.
    
    Parameters:
    ensemble: (16, 20) array of ensemble members
    observations: (10,) array of observed values (even indices)
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    updated_ensemble: (16, 20) array
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n_members, n_vars = ensemble.shape
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    
    # 1. Compute ensemble mean and anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean  # (16, 20)
    
    # 2. Compute background error covariance P (20x20)
    # P = (1/(N-1)) * A A^T
    # But we only need the update for the full state based on observations.
    # Let A_obs be anomalies at observed indices (16, 10)
    # Let A_full be all anomalies (16, 20)
    
    A_obs = anomalies[:, obs_idx]  # (16, 10)
    
    # 3. Compute innovation
    # y = observations (10,)
    # y_hat = mean[obs_idx] (10,)
    y_hat = mean[obs_idx]
    innovation = observations - y_hat  # (10,)
    
    # 4. Compute Kalman Gain
    # S = P_yy + R
    # P_yy = (1/(N-1)) * A_obs A_obs^T (10x10)
    P_yy = (A_obs @ A_obs.T) / (n_members - 1)
    
    R = observation_variance * np.eye(n_obs)
    S = P_yy + R
    
    # K = P_xy S^-1
    # P_xy = (1/(N-1)) * A_full A_obs^T (20x10)
    P_xy = (anomalies @ A_obs.T) / (n_members - 1)
    
    # Localization: Apply localization weights to P_xy
    # L is (20, 10)
    L = _localization_matrix(obs_idx, n_vars)
    P_xy_loc = P_xy * L
    
    # Solve S K^T = P_xy_loc  => K = (S^-1 P_xy_loc^T)^T
    # More efficiently: K = P_xy_loc @ inv(S)^T
    # Since S is symmetric, inv(S)^T = inv(S)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    K = P_xy_loc @ S_inv  # (20, 10)
    
    # 5. Update mean
    mean_new = mean + K @ innovation
    
    # 6. Update ensemble members (perturbation update)
    # Standard perturbation update:
    # x_i' = x_i + K (y - y_hat_i)
    # where y_hat_i = mean_i[obs_idx]
    
    updated_ensemble = ensemble.copy()
    
    # Compute predicted observations for each member
    # y_hat_members = ensemble[:, obs_idx] (16, 10)
    y_hat_members = ensemble[:, obs_idx]
    
    # Innovation for each member
    # innov_members = observations[None, :] - y_hat_members (16, 10)
    innov_members = observations[None, :] - y_hat_members
    
    # Update: x_i' = x_i + K @ innov_members[i]
    # K is (20, 10). innov_members is (16, 10).
    # We need to compute (16, 10) @ (10, 20) ? No, K is (20,10).
    # Update vector for member i is K @ innov_members[i] (20,)
    # So updated_ensemble = ensemble + innov_members @ K.T
    
    updated_ensemble += innov_members @ K.T
    
    # 7. Adaptive Inflation
    # Check if the ensemble spread has collapsed or if the observation misfit is large relative to spread.
    # Simple heuristic: If the ratio of observed innovation variance to S is too high, inflate.
    # Or, maintain a minimum spread.
    
    # Calculate spread at observed points
    spread_obs = np.std(updated_ensemble[:, obs_idx], axis=0)
    mean_spread = np.mean(spread_obs)
    
    # Target spread: roughly sqrt(observation_variance) or a fraction of state range
    # In L96, variables are O(10). Obs var is 0.49 -> std 0.7.
    # Let's target a spread of at least 1.0 or so.
    target_spread = 1.5
    
    if mean_spread < target_spread:
        # Inflate around the mean
        inflation_factor = target_spread / (mean_spread + 1e-6)
        # Cap inflation to avoid instability
        inflation_factor = min(inflation_factor, 1.5)
        
        updated_ensemble = mean_new + (updated_ensemble - mean_new) * inflation_factor
    
    # Ensure finiteness and bounds
    updated_ensemble[~np.isfinite(updated_ensemble)] = mean_new[~np.isfinite(mean_new)]
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    
    # Check deadline
    if time.time() > deadline:
        return updated_ensemble
        
    return updated_ensemble

# EVOLVE-BLOCK-END
