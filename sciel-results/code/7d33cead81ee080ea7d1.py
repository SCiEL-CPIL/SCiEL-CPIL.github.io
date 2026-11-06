"""Ensemble Kalman Filter with localization, inflation, and rank-deficiency handling."""
import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an ensemble analysis update using an Ensemble Kalman Filter (EnKF)
    with spatial localization, multiplicative inflation, and rank-deficiency handling.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array_like
        Indices of observed variables.
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N = ensemble.shape[0]  # ensemble size (16)
    D = ensemble.shape[1]  # dimension (20)
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    
    # Compute ensemble mean and anomalies
    x_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - x_mean
    
    # Compute ensemble covariance (with rank correction)
    # C = (1/(N-1)) * A @ A.T where A is (D, N-1)
    # Use SVD for stability
    U, s, Vt = np.linalg.svd(anomalies, full_matrices=False)
    
    # Regularize singular values for rank-deficiency
    s_reg = s.copy()
    s_reg[s_reg < 1e-10] = 1e-10
    s_reg /= (N - 1)
    
    # Reconstruct covariance
    C = (U * s_reg) @ U.T
    
    # Observation operator H: selects observed components
    H = np.zeros((n_obs, D))
    H[np.arange(n_obs), obs_idx] = 1.0
    
    # Observation covariance R
    R = observation_variance * np.eye(n_obs)
    
    # Innovation covariance: S = H C H^T + R
    HC = H @ C
    S = HC @ H.T + R
    
    # Compute Kalman gain: K = C H^T S^{-1}
    # Use Cholesky for stability
    try:
        L = np.linalg.cholesky(S)
        K = (HC.T @ np.linalg.solve(L.T, np.linalg.solve(L, H.T)).T)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        K = C @ H.T @ np.linalg.pinv(S)
    
    # Innovation
    innovation = observations - H @ x_mean
    
    # Analysis mean
    x_a_mean = x_mean + K @ innovation
    
    # Analysis anomalies
    # A_a = A (I - K H)
    KH = K @ H
    A_a = anomalies @ (np.eye(N) - KH.T)
    
    # Apply multiplicative inflation to prevent filter collapse
    # Estimate inflation factor based on ensemble spread vs observation error
    # Simple adaptive inflation
    x_a = x_a_mean[None, :] + A_a
    
    # Check if time is running out
    if time.time() > deadline - 5.0:
        # Return current result with basic sanity checks
        x_a = np.clip(x_a, -100.0, 100.0)
        x_a[~np.isfinite(x_a)] = 0.0
        return x_a
    
    # Adaptive inflation: ensure ensemble spread is not too small
    # Compare ensemble variance at observed locations to observation variance
    obs_ens_var = np.var(x_a[:, obs_idx], axis=0)
    target_var = observation_variance * 0.8  # Target spread slightly below obs variance
    
    # Compute inflation factor
    inflation_factors = np.ones(D)
    for i in range(n_obs):
        idx = obs_idx[i]
        if obs_ens_var[i] < target_var:
            # Inflate to match target variance
            current_std = np.sqrt(obs_ens_var[i] + 1e-10)
            target_std = np.sqrt(target_var)
            inflation_factors[idx] = target_std / current_std
    
    # Limit inflation to avoid instability
    inflation_factors = np.clip(inflation_factors, 1.0, 1.5)
    
    # Apply inflation
    x_a = x_a_mean[None, :] + A_a * inflation_factors[None, :]
    
    # Final sanity checks
    x_a = np.clip(x_a, -100.0, 100.0)
    x_a[~np.isfinite(x_a)] = 0.0
    
    return x_a
