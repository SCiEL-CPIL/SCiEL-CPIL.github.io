"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with localization, inflation, and outlier rejection.
    
    Parameters:
    -----------
    ensemble : np.ndarray of shape (16, 20)
        Current ensemble members.
    observations : np.ndarray of shape (16, 10)
        Noisy observations of the even-indexed states.
    observed_indices : array-like of shape (10,)
        Indices of the observed variables (even indices: 0, 2, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : np.ndarray of shape (16, 20)
        Updated ensemble after the analysis step.
    """
    num_members, num_vars = ensemble.shape
    obs_idx = np.asarray(observed_indices)
    num_obs = len(obs_idx)
    
    # Copy ensemble to avoid modifying input
    X = ensemble.copy().astype(np.float64)
    y = observations.astype(np.float64)
    
    # Compute ensemble mean and covariance
    X_mean = X.mean(axis=0)
    X_anom = X - X_mean
    
    # Compute observation operator (just selection of observed indices)
    y_mean = y.mean(axis=0)
    y_anom = y - y_mean
    
    # Observation covariance (diagonal)
    R = observation_variance * np.eye(num_obs)
    
    # Compute cross-covariance between state and observations
    # C_xy = X_anom[:, obs_idx] @ y_anom.T / (num_members - 1)
    X_obs_anom = X_anom[:, obs_idx]
    C_xy = (X_obs_anom @ y_anom.T) / (num_members - 1)
    
    # Compute observation covariance from ensemble + measurement error
    # C_yy = X_obs_anom @ X_obs_anom.T / (num_members - 1) + R
    C_yy = (X_obs_anom @ X_obs_anom.T) / (num_members - 1) + R
    
    # Add small regularization for numerical stability
    C_yy += 1e-10 * np.eye(num_obs)
    
    # Compute Kalman gain
    # K = C_xy @ inv(C_yy)
    try:
        C_yy_inv = np.linalg.inv(C_yy)
        K = C_xy @ C_yy_inv
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        C_yy_inv = np.linalg.pinv(C_yy)
        K = C_xy @ C_yy_inv
    
    # Innovation: y - H(X_mean)
    innovation = y_mean - X_mean[obs_idx]
    
    # Update ensemble mean
    X_mean_new = X_mean + K @ innovation
    
    # Update ensemble anomalies
    # X_anom_new = X_anom - K @ (H(X_anom) - y_anom)
    # H(X_anom) = X_anom[:, obs_idx]
    # So X_anom_new = X_anom - K @ (X_obs_anom - y_anom)
    X_anom_new = X_anom - K @ (X_obs_anom - y_anom)
    
    # Reconstruct updated ensemble
    X_new = X_mean_new + X_anom_new
    
    # Apply multiplicative inflation to prevent filter collapse
    # Estimate inflation factor based on spread vs observation consistency
    # Simple adaptive inflation: if ensemble spread is too small relative to obs error, inflate
    spread_obs = np.std(X_new[:, obs_idx], axis=0)
    obs_error = np.sqrt(observation_variance)
    
    # Target spread should be comparable to observation error
    # Use a simple inflation factor
    inflation_factor = 1.0
    for i in range(num_obs):
        if spread_obs[i] < 0.5 * obs_error:
            inflation_factor = max(inflation_factor, 1.05)
    
    # Apply inflation around the mean
    if inflation_factor > 1.0:
        X_new = X_mean_new + inflation_factor * (X_new - X_mean_new)
    
    # Clip to reasonable bounds to ensure finiteness and within [-100, 100]
    X_new = np.clip(X_new, -100.0, 100.0)
    
    # Ensure no NaN or Inf
    X_new = np.nan_to_num(X_new, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return X_new
# EVOLVE-BLOCK-END
