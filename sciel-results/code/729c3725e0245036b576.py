"""Ensemble Kalman filter for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman filter analysis update for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n_members, n_state = ensemble.shape
    n_obs = len(observed_indices)
    
    # Convert to float64 for numerical stability
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64)
    obs_idx = observed_indices.astype(int)
    obs_var = float(observation_variance)
    
    # Compute ensemble mean and covariance
    ens_mean = np.mean(ens, axis=0)
    
    # Compute anomaly matrix: (n_members, n_state)
    anomalies = ens - ens_mean
    
    # Compute sample covariance: (n_state, n_state)
    P = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Observation operator: H maps state to observation space
    # H is sparse: H[i, obs_idx[i]] = 1
    # Observation covariance: R = obs_var * I
    R = obs_var * np.eye(n_obs)
    
    # Innovation covariance: S = H P H^T + R
    # H P H^T is (n_obs, n_obs)
    HP = P[np.ix_(obs_idx, obs_idx)]
    S = HP + R
    
    # Compute Kalman gain: K = P H^T S^{-1}
    # P H^T is (n_state, n_obs)
    PHt = P[:, obs_idx]
    
    # Solve S^{-1} * PHt^T instead of inverting S
    # K = PHt @ S^{-1}
    try:
        S_inv_PHt = np.linalg.solve(S, PHt.T)
    except np.linalg.LinAlgError:
        S_inv_PHt = np.linalg.lstsq(S, PHt.T, rcond=None)[0]
    
    K = (S_inv_PHt.T)  # (n_state, n_obs)
    
    # Innovation: y - H x_mean
    innovation = obs - ens_mean[obs_idx]
    
    # Update mean: x_mean + K * innovation
    new_mean = ens_mean + K @ innovation
    
    # Update anomalies: a_i - K * (H a_i)
    # H a_i = a_i[obs_idx]
    ens_anoms = ens - ens_mean  # (n_members, n_state)
    H_anoms = ens_anoms[:, obs_idx]  # (n_members, n_obs)
    
    # Update each member
    # new_ens[i] = new_mean + (anoms[i] - K @ H @ anoms[i])
    updated_anoms = ens_anoms - (K @ H_anoms.T).T  # (n_members, n_state)
    new_ens = new_mean + updated_anoms
    
    # Ensure finite values
    if not np.all(np.isfinite(new_ens)):
        # Fallback: use original ensemble with simple nudging
        result = ensemble.copy()
        result[:, obs_idx] += 0.6 * (obs - result[:, obs_idx])
        return result
    
    # Clip to [-100, 100]
    new_ens = np.clip(new_ens, -100.0, 100.0)
    
    # Check deadline
    if time.time() - t0 > deadline:
        return new_ens.astype(np.float64)
    
    return new_ens.astype(np.float64)
# EVOLVE-BLOCK-END
