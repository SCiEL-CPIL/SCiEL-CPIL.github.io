"""Ensemble Square Root Filter with sequential coordinate-wise regression for Lorenz-96 assimilation."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Square Root Filter (EnSRF) analysis update using
    sequential, coordinate-wise regression of unobserved variables onto observed ones.
    
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
    
    # Ensure observed_indices is a numpy array
    obs_idx = np.asarray(observed_indices, dtype=int)
    obs = np.asarray(observations, dtype=np.float64)
    R = float(observation_variance)
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean
    
    # We'll do a sequential update: first update the observed coordinates,
    # then use regression to update unobserved coordinates based on the
    # updated observed coordinates.
    
    # Step 1: Update observed coordinates using an EnSRF-style adjustment
    # For observed coordinates, we blend with the observation using the 
    # ensemble variance and observation variance.
    
    obs_var_ens = np.var(anomalies[:, obs_idx], axis=0)  # shape (n_obs,)
    
    # Kalman gain for observed coordinates
    K_obs = obs_var_ens / (obs_var_ens + R)  # shape (n_obs,)
    
    # Update observed coordinates in each ensemble member
    # The observation residual for the mean
    obs_resid = obs - ens_mean[obs_idx]
    
    # Adjust the mean of observed coordinates
    ens_mean[obs_idx] += K_obs * obs_resid
    
    # Adjust anomalies of observed coordinates (scale by sqrt(1-K))
    scale_obs = np.sqrt(1.0 - K_obs)  # shape (n_obs,)
    anomalies[:, obs_idx] *= scale_obs[np.newaxis, :]
    
    # Reconstruct ensemble with updated observed coordinates
    ens = ens_mean[np.newaxis, :] + anomalies
    
    # Step 2: Update unobserved coordinates via regression onto observed coordinates
    # For each unobserved variable j, we regress it onto the observed variables
    # using the ensemble covariance.
    
    unobs_idx = np.array([i for i in range(n_state) if i not in set(obs_idx.tolist())])
    
    if len(unobs_idx) == 0:
        return ens
    
    # Compute ensemble covariance between observed and unobserved variables
    # Using the anomalies (centered data)
    # C_oo = cov(obs, obs), C_ou = cov(obs, unobs), C_uu = cov(unobs, unobs)
    
    # Use sample covariance (with n-1 normalization for better estimation)
    n_obs_vars = len(obs_idx)
    n_unobs_vars = len(unobs_idx)
    
    # Covariance matrices
    C_oo = (anomalies[:, obs_idx].T @ anomalies[:, obs_idx]) / (n_ens - 1)
    C_ou = (anomalies[:, obs_idx].T @ anomalies[:, unobs_idx]) / (n_ens - 1)
    C_uu = (anomalies[:, unobs_idx].T @ anomalies[:, unobs_idx]) / (n_ens - 1)
    
    # For each unobserved variable, compute the regression coefficients
    # beta = C_uu_obs^{-1} * C_ou  (for each unobserved var)
    # Actually, for variable j (unobserved), the regression onto observed vars:
    # beta_j = C_oo^{-1} * C_ou[:, j]  where C_ou[:, j] is cov(obs, unobs_j)
    
    # But we need to account for the fact that the observed variables have already
    # been updated. The standard approach in sequential EnSRF is to use the 
    # current ensemble statistics.
    
    # Compute the Kalman gain for unobserved variables
    # K_unobs = C_ou * C_oo^{-1}  (shape: n_unobs x n_obs)
    
    # Use a regularized inverse to avoid numerical issues
    # C_oo is 10x10, should be well-conditioned for Lorenz-96
    try:
        C_oo_inv = np.linalg.inv(C_oo)
    except np.linalg.LinAlgError:
        C_oo_inv = np.linalg.pinv(C_oo)
    
    # K_unobs[j, k] = sum_m C_ou[j, m] * C_oo_inv[m, k]
    K_unobs = C_ou @ C_oo_inv  # shape (n_unobs, n_obs)
    
    # The observation residual (same as before)
    # obs_resid is already computed: obs - original_mean[obs_idx]
    # But we need the residual relative to the pre-update mean for the regression
    # Actually, in the sequential approach, we use the updated observed anomalies
    # to update the unobserved variables.
    
    # The change in the observed mean is: delta_mean_obs = K_obs * obs_resid
    # The new observed anomalies are scaled by sqrt(1-K_obs)
    
    # For the unobserved variables, the update is:
    # new_unobs_anomaly = old_unobs_anomaly + K_unobs @ (new_obs_anomaly - old_obs_anomaly)
    # But new_obs_anomaly = sqrt(1-K_obs) * old_obs_anomaly
    # So new_obs_anomaly - old_obs_anomaly = (sqrt(1-K_obs) - 1) * old_obs_anomaly
    
    # Also, the mean of unobserved variables should be updated by:
    # delta_mean_unobs = K_unobs @ delta_mean_obs
    
    # Update unobserved means
    delta_mean_obs = K_obs * obs_resid  # shape (n_obs,)
    delta_mean_unobs = K_unobs @ delta_mean_obs  # shape (n_unobs,)
    ens_mean[unobs_idx] += delta_mean_unobs
    
    # Update unobserved anomalies
    # The change in observed anomalies: (sqrt(1-K_obs) - 1) * old_obs_anomalies
    delta_obs_anom = (scale_obs - 1.0) * anomalies[:, obs_idx]  # shape (n_ens, n_obs)
    delta_unobs_anom = delta_obs_anom @ K_unobs.T  # shape (n_ens, n_unobs)
    anomalies[:, unobs_idx] += delta_unobs_anom
    
    # Reconstruct final ensemble
    ens = ens_mean[np.newaxis, :] + anomalies
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finite values
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    return ens
# EVOLVE-BLOCK-END
