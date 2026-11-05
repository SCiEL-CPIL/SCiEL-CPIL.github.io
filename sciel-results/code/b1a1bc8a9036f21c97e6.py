"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with localization and inflation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (16, 10) or similar
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
    
    # Handle case where observations shape might be (n_obs,) for a single snapshot
    # but the task says ensemble is (16,20) and we observe even-indexed states.
    # The observations argument likely has shape (n_members, n_obs) or (n_obs,).
    # Let's assume standard EnKF setup where observations are for each member or a single truth.
    # Given the interface `analyze(ensemble, observations, ...)`, and ensemble is (16,20),
    # observations is likely (16, 10) representing noisy obs for each member? 
    # No, typically in data assimilation, we have one set of observations (from truth) 
    # and an ensemble. But the prompt says "noisy observations of half its coordinates".
    # Let's look at the seed: `result[:, observed_indices] += 0.6 * (observations - result[:, observed_indices])`
    # This implies `observations` has the same shape as `ensemble[:, observed_indices]`, i.e., (16, 10).
    # This is a bit unusual for standard EnKF which uses a single observation vector.
    # However, if each member has its own "observation" (which might just be the same noisy obs repeated, 
    # or perhaps it's an ensemble of observations?), the seed treats them independently.
    # Actually, in many assimilation setups, `observations` is a single vector of length n_obs.
    # But the seed code does `result[:, observed_indices]` which is (16, 10) and adds `0.6 * (observations - ...)`.
    # For this to work, `observations` must be broadcastable to (16, 10).
    # If `observations` is (10,), it broadcasts. If it's (16, 10), it works element-wise.
    # Let's assume `observations` is either (n_obs,) or (n_members, n_obs).
    
    # To be safe and robust, let's check the shape.
    if observations.ndim == 1:
        obs_vec = observations
        obs_shape = (n_obs,)
    else:
        obs_vec = observations
        obs_shape = observations.shape
    
    # We will implement a standard EnKF update.
    # The state vector is x (20 variables).
    # We observe y (10 variables).
    
    # Compute ensemble mean
    x_mean = np.mean(ensemble, axis=0)
    
    # Compute ensemble covariance
    x_anom = ensemble - x_mean
    # P = (1/(n-1)) * X_anom * X_anom.T
    P = (x_anom.T @ x_anom) / (n_members - 1)
    
    # Observation operator H: selects observed indices
    # H is (n_obs, n_vars)
    H = np.zeros((n_obs, n_vars))
    H[np.arange(n_obs), observed_indices] = 1.0
    
    # Observation covariance R
    R = np.eye(n_obs) * observation_variance
    
    # Kalman gain K = P H^T (H P H^T + R)^-1
    # HP = P H^T (n_vars x n_obs) -> actually P is (n_vars, n_vars), H^T is (n_vars, n_obs)
    # So P H^T is (n_vars, n_obs)
    PHt = P @ H.T
    # H P H^T + R is (n_obs, n_obs)
    HPHtR = H @ PHt + R
    
    # Solve for gain
    try:
        K = PHt @ np.linalg.inv(HPHtR)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse or regularization
        K = PHt @ np.linalg.pinv(HPHtR)
    
    # Innovation: y - H x_mean
    # If observations is a single vector (n_obs,), use it directly.
    # If observations is (n_members, n_obs), we have a problem: standard EnKF uses one y.
    # The seed code suggests element-wise update, which is not standard EnKF.
    # However, if the evaluator passes the same noisy observation for all members (broadcasted),
    # then we can just use the mean or any single row.
    # Let's assume the most common case: `observations` is the single observation vector (n_obs,).
    # If it's (16, 10), it's likely the same obs repeated or independent obs per member?
    # Given "noisy observations of half its coordinates" and "independent N(0,0.49) noise",
    # it's likely a single observation vector from the truth.
    # Let's handle both cases.
    
    if observations.ndim == 1:
        y = observations
    else:
        # If 2D, take the mean across members to get a single observation vector?
        # Or assume it's the same for all members and take the first row.
        y = observations[0]
    
    innovation = y - H @ x_mean
    
    # Update mean
    x_mean_new = x_mean + K @ innovation
    
    # Update ensemble members
    # x_i_new = x_i + K (y - H x_i)  <-- This is the "stochastic" EnKF or "deterministic" if using mean innovation?
    # Standard deterministic EnKF:
    # x_i_new = x_i + K (y - H x_mean)  <-- This is not quite right for ensemble spread.
    # Correct deterministic EnKF:
    # x_i_new = x_i + K (y - H x_i)  is the stochastic version (using same y for all).
    # But to preserve the ensemble mean exactly, we use:
    # x_i_new = x_i + K (y - H x_mean)  ... no, this doesn't account for individual anomalies.
    
    # Let's use the standard deterministic EnKF formula:
    # x_i_new = x_i + K (y - H x_i)
    # This changes the mean slightly if K is not symmetric, but it's the standard "local" update.
    # Alternatively, the "EnKF" by Evensen uses:
    # x_i_new = x_i + K (y - H x_i)
    
    # Let's use the standard form:
    # x_i_new = x_i + K @ (y - H @ x_i)
    
    updated_ensemble = ensemble.copy()
    for i in range(n_members):
        Hx_i = H @ ensemble[i]
        innov_i = y - Hx_i
        updated_ensemble[i] = ensemble[i] + K @ innov_i
    
    # Optional: Inflation to prevent filter collapse
    # Inflate the ensemble around the new mean
    inflation_factor = 1.02  # Slight inflation
    updated_ensemble = inflation_factor * (updated_ensemble - np.mean(updated_ensemble, axis=0)) + np.mean(updated_ensemble, axis=0)
    
    # Localization: Reduce the impact of distant observations
    # Since we have a 1D lattice (Lorenz-96), we can apply a localization function to K.
    # K is (n_vars, n_obs). We want to reduce K[i, j] if |i - observed_indices[j]| is large.
    # Simple exponential localization:
    # K[i, j] *= exp(-|i - obs_idx_j|^2 / (2 * L^2))
    # Let's apply this to K before the update? We already did the update.
    # Let's re-do it with localization.
    
    # Recompute K with localization
    K_loc = K.copy()
    L = 3.0  # Localization length scale
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_vars):
            dist = abs(i - obs_idx)
            K_loc[i, j] *= np.exp(-dist**2 / (2 * L**2))
    
    # Redo the update with localized K
    updated_ensemble = ensemble.copy()
    for i in range(n_members):
        Hx_i = H @ ensemble[i]
        innov_i = y - Hx_i
        updated_ensemble[i] = ensemble[i] + K_loc @ innov_i
    
    # Re-apply inflation
    mean_new = np.mean(updated_ensemble, axis=0)
    updated_ensemble = inflation_factor * (updated_ensemble - mean_new) + mean_new
    
    # Ensure finite values
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    # Clip to [-100, 100] as per contract
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
