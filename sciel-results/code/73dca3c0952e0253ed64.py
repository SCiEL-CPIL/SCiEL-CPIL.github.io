"""EnSRF-based assimilation for Lorenz-96 with localization and adaptive inflation."""
import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Nonlinear ensemble analysis using Ensemble Square Root Filter (EnSRF) with
    local linear regression, localization, and adaptive inflation.
    
    Parameters:
        ensemble: (16, 20) array of ensemble members
        observations: (10,) array of noisy observations of even-indexed states
        observed_indices: array of observed state indices (even indices)
        observation_variance: scalar, variance of observation noise
    
    Returns:
        (16, 20) updated ensemble
    """
    n_members, n_states = ensemble.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # Start with a copy
    result = ensemble.copy()
    
    # --- Adaptive Inflation ---
    # Compute ensemble spread at observed locations
    ens_mean_obs = result[:, obs_idx].mean(axis=0)
    ens_spread_obs = np.sqrt(np.mean((result[:, obs_idx] - ens_mean_obs)**2, axis=0))
    
    # Innovation variance
    innovation = observations - ens_mean_obs
    innov_var = np.mean(innovation**2)
    
    # Expected innovation variance
    expected_innov_var = np.mean(ens_spread_obs**2) + observation_variance
    
    # Inflation factor
    if expected_innov_var > 1e-12:
        ratio = innov_var / expected_innov_var
        # Adaptive inflation: slightly inflate if spread is too small
        if ratio < 1.0:
            inflation = 1.0 + 0.05 * (1.0 - ratio)
        else:
            inflation = 1.0 + 0.02 * (ratio - 1.0)
        inflation = np.clip(inflation, 0.95, 1.15)
    else:
        inflation = 1.0
    
    # Apply inflation about the mean
    global_mean = result.mean(axis=0)
    result = global_mean + inflation * (result - global_mean)
    
    # --- Localization for EnSRF regression ---
    # Use a Gaussian localization kernel based on ring distance
    def ring_dist(i, j, n):
        d = abs(i - j)
        return min(d, n - d)
    
    # Localization length scale (in grid points)
    loc_scale = 4.0
    
    # --- EnSRF update via local linear regression ---
    # For each unobserved state, regress it on the observed states using ensemble perturbations
    # This is a nonlinear mapping that accounts for error growth
    
    # Center the ensemble
    ens_mean = result.mean(axis=0)
    perturbations = result - ens_mean  # (16, 20)
    
    # Observed perturbations
    obs_pert = perturbations[:, obs_idx]  # (16, 10)
    obs_mean = ens_mean[obs_idx]
    
    # Compute regression coefficients for each unobserved state
    # We want to find B such that perturbations[:, i] ≈ B @ obs_pert
    # Using least squares: B = (obs_pert^T @ obs_pert)^{-1} @ obs_pert^T @ perturbations[:, i]
    
    # Add regularization for numerical stability
    reg = 1e-6 * np.eye(n_obs)
    obs_pert_T_obs_pert = obs_pert.T @ obs_pert + reg  # (10, 10)
    
    try:
        obs_pert_T_obs_pert_inv = np.linalg.inv(obs_pert_T_obs_pert)
    except np.linalg.LinAlgError:
        obs_pert_T_obs_pert_inv = np.linalg.pinv(obs_pert_T_obs_pert)
    
    # For each state, compute the regression coefficient
    # B[i] = obs_pert_T_obs_pert_inv @ (obs_pert.T @ perturbations[:, i])
    # But we also need to apply localization
    
    # Compute the full regression matrix
    # B_full[i, j] = coefficient for state i regressed on observed state j
    # perturbations is (16, 20), obs_pert is (16, 10)
    
    # obs_pert.T @ perturbations is (10, 20)
    cross_cov = obs_pert.T @ perturbations  # (10, 20)
    B_full = obs_pert_T_obs_pert_inv @ cross_cov  # (10, 20)
    
    # Apply localization to the regression coefficients
    # For each state i, weight the contribution of each observed state j by localization
    for i in range(n_states):
        if i in obs_idx:
            continue
        weights = np.array([np.exp(-0.5 * (ring_dist(i, j, n_states) / loc_scale)**2) 
                           for j in range(n_obs)])
        # Localized B for state i
        B_loc = B_full[:, i] * weights  # (10,)
        
        # The regression-based estimate of the perturbation for state i
        # pert_i ≈ B_loc @ obs_pert
        # But we need to combine this with the observation update
        
        # For EnSRF, the update is:
        # x_i_new = x_i_old + B_loc @ (obs - obs_mean)
        # where obs - obs_mean is the innovation
        
        innovation_vec = observations - obs_mean  # (10,)
        correction = B_loc @ innovation_vec
        result[:, i] += correction
    
    # --- Observation update for observed states ---
    # For observed states, use a Kalman-like update
    # But with the nonlinear regression, we can use a simpler approach
    
    # Compute the gain for observed states
    # For each observed state, the variance is the ensemble variance at that location
    obs_var = np.var(result[:, obs_idx], axis=0, ddof=1)
    obs_var = np.maximum(obs_var, 1e-6)
    
    # Kalman gain
    K = obs_var / (obs_var + observation_variance)
    
    # Apply observation update
    for k, idx in enumerate(obs_idx):
        innovation_k = observations[k] - result[:, idx].mean()
        result[:, idx] += K[k] * innovation_k
    
    # --- Re-center to maintain ensemble mean consistency ---
    # The above updates may have shifted the mean; re-center if needed
    # Actually, the updates should preserve the mean structure
    
    # Clip to valid range
    result = np.clip(result, -100, 100)
    
    # Ensure finiteness
    if not np.all(np.isfinite(result)):
        result = ensemble.copy()
        result[:, obs_idx] += 0.5 * (observations - result[:, obs_idx])
    
    return result
