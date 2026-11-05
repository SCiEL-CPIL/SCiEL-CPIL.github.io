"""Ensemble Square Root Filter with regression-based update for Lorenz-96 assimilation."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Square Root Filter (EnSRF) analysis update.
    
    1. Update observed coordinates using local Kalman gains.
    2. Update unobserved coordinates using linear regression onto the 
       observed coordinates, based on the forecast covariance structure.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of the observed coordinates (even-indexed).
    observed_indices : array-like
        Indices of the observed coordinates.
    observation_variance : float
        Variance of the observation noise.
        
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    ens = ensemble.copy().astype(np.float64)
    n_ens, n_state = ens.shape
    
    obs = np.asarray(observations, dtype=np.float64)
    obs_idx_arr = np.asarray(observed_indices, dtype=int)
    R = float(observation_variance)
    
    # Build a set of observed indices for quick lookup
    obs_idx_set = set(obs_idx_arr.tolist())
    
    # Store the forecast ensemble for regression calculations
    X_forecast = ens.copy()
    
    # Initialize the analysis ensemble
    X_analysis = ens.copy()
    
    # --- Pass 1: Update Observed Coordinates ---
    for k, idx in enumerate(obs_idx_arr):
        idx = int(idx)
        x_j = X_forecast[:, idx]
        mu_j = np.mean(x_j)
        a_j = x_j - mu_j
        var_j = np.var(a_j)
        
        if var_j < 1e-12:
            # Degenerate case: set directly to observation
            X_analysis[:, idx] = obs[k]
            continue
        
        # Kalman gain
        K_j = var_j / (var_j + R)
        
        # Observation residual
        resid = obs[k] - mu_j
        
        # Update mean
        mu_j_new = mu_j + K_j * resid
        
        # Scale anomalies (reduces variance)
        scale_j = np.sqrt(1.0 - K_j)
        a_j_new = a_j * scale_j
        
        X_analysis[:, idx] = mu_j_new + a_j_new

    # --- Pass 2: Update Unobserved Coordinates via Regression ---
    # We regress the unobserved variables onto the observed variables.
    # The relationship is derived from the forecast ensemble.
    # Delta x_unobs = beta^T * Delta x_obs
    # where Delta x = x_analysis - x_forecast
    
    # Prepare observed data for regression
    # We use anomalies for the regression coefficients to avoid intercept issues
    # but the delta includes the mean shift.
    
    obs_cols_forecast = X_forecast[:, obs_idx_arr]
    obs_cols_analysis = X_analysis[:, obs_idx_arr]
    
    # Calculate deltas (analysis - forecast) for observed variables
    delta_obs = obs_cols_analysis - obs_cols_forecast  # shape (n_ens, n_obs)
    
    # Center the forecast observed anomalies for regression
    obs_mean_forecast = np.mean(obs_cols_forecast, axis=0)
    obs_anom_forecast = obs_cols_forecast - obs_mean_forecast[np.newaxis, :]
    
    # Compute regression coefficients for all unobserved variables at once
    # We want to solve: A_obs @ B = A_unobs
    # where A_obs is (n_ens, n_obs), A_unobs is (n_ens, n_unobs)
    # B is (n_obs, n_unobs)
    
    unobs_indices = [j for j in range(n_state) if j not in obs_idx_set]
    
    if unobs_indices:
        unobs_cols_forecast = X_forecast[:, unobs_indices]
        unobs_mean_forecast = np.mean(unobs_cols_forecast, axis=0)
        unobs_anom_forecast = unobs_cols_forecast - unobs_mean_forecast[np.newaxis, :]
        
        # Solve for regression coefficients B
        # Using lstsq for stability
        try:
            B, _, _, _ = np.linalg.lstsq(obs_anom_forecast, unobs_anom_forecast, rcond=None)
        except np.linalg.LinAlgError:
            B = np.zeros((len(obs_idx_arr), len(unobs_indices)))
        
        # The correction for unobserved variables is:
        # Delta x_unobs = Delta x_obs @ B
        # Note: Delta x_obs includes the mean shift and the anomaly change.
        # Since the regression was fit on anomalies, does it apply to the full delta?
        # Yes, because E[Delta x_unobs | Delta x_obs] is linear.
        # The regression B minimizes || obs_anom @ B - unobs_anom ||^2.
        # The optimal linear predictor for the *changes* is the same B if the 
        # joint distribution is Gaussian and the covariance structure is preserved.
        
        delta_unobs = delta_obs @ B  # shape (n_ens, n_unobs)
        
        # Apply the correction
        for i, idx in enumerate(unobs_indices):
            X_analysis[:, idx] += delta_unobs[:, i]
    
    # Clip to valid range and ensure finiteness
    X_analysis = np.clip(X_analysis, -100.0, 100.0)
    X_analysis = np.where(np.isfinite(X_analysis), X_analysis, 0.0)
    
    return X_analysis
# EVOLVE-BLOCK-END
