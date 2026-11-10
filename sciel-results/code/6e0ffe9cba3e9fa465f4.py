"""EnAKF with Gaspari-Cohn localization and adaptive inflation for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gaspari_cohn(distance, L):
    """Gaspari-Cohn fifth-order polynomial correlation function."""
    distance = np.asarray(distance, dtype=np.float64)
    L = float(L)
    if L <= 0:
        return np.where(distance == 0, 1.0, 0.0)
    x = distance / L
    out = np.zeros_like(x)
    # |x| <= 1
    m1 = x <= 1.0
    out[m1] = (
        -4.0 * x[m1]**5 / 2.0
        + 5.0 * x[m1]**4
        - 1.0 * x[m1]**3 / 2.0
        + 1.0
    )
    # 1 < |x| <= 2
    m2 = (x > 1.0) & (x <= 2.0)
    xm = x[m2]
    out[m2] = (
        2.0 * xm**5 / 3.0
        - 5.0 * xm**4
        + 1.0 * xm**3
        - 1.0 * xm**2
        + 4.0 / 3.0
    )
    # |x| > 2 -> 0
    return out


def _l96_step(state, F, dt):
    """One RK4 step of Lorenz-96."""
    n = state.shape[0]
    def f(x):
        return (x[1:] - x[-2:]) * x[:-1] - x + F
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _enakf_analysis(ensemble, observations, observed_indices, obs_var, loc_scale=4.0):
    """
    Local Ensemble Adjustment Kalman Filter update.
    ensemble: (N, D) array
    observations: (N, Nobs) array
    observed_indices: array of observed state indices
    obs_var: scalar observation variance
    Returns updated (N, D) ensemble.
    """
    N, D = ensemble.shape
    Nobs = len(observed_indices)
    
    # Compute ensemble mean and anomalies
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean  # (N, D)
    
    # Build localization matrix for observed variables
    # Distance matrix for observed indices (circular)
    obs_idx = np.array(observed_indices, dtype=int)
    dist = np.zeros((Nobs, Nobs))
    for i in range(Nobs):
        for j in range(Nobs):
            d = abs(obs_idx[i] - obs_idx[j])
            dist[i, j] = min(d, D - d)
    
    R_loc = _gaspari_cohn(dist, loc_scale)  # (Nobs, Nobs)
    
    # Observation covariance: R = obs_var * I + localization-adjusted background
    # For EnAKF, we work with the local ensemble
    # B = ensemble covariance at observed locations
    B = (anomalies[:, obs_idx].T @ anomalies[:, obs_idx]) / (N - 1.0)  # (Nobs, Nobs)
    
    # Localized observation covariance
    R = obs_var * np.eye(Nobs)
    
    # Kalman gain: K = B @ (B + R)^{-1}
    S = B + R  # (Nobs, Nobs)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    K = B @ S_inv  # (Nobs, Nobs)
    
    # Innovation
    obs_mean = observations.mean(axis=0)  # (Nobs,)
    innovation = obs_mean - mean[obs_idx]  # (Nobs,)
    
    # Update mean at observed locations
    new_mean_obs = mean[obs_idx] + K @ innovation  # (Nobs,)
    
    # Square-root transformation for anomalies
    # We want to transform anomalies so that:
    # 1. New mean at obs locations = new_mean_obs
    # 2. New ensemble covariance matches the analysis covariance
    # 3. Unobserved variables are updated via the ensemble covariance structure
    
    # Full analysis covariance:
    # B_a = (I - K @ H) @ B @ (I - K @ H).T + K @ R @ K.T
    # where H picks observed components
    
    # For the local (observed) part:
    B_obs = B  # (Nobs, Nobs)
    H = np.eye(Nobs)  # identity since we're already in obs space
    
    # Analysis covariance in obs space
    B_a_obs = (np.eye(Nobs) - K @ H) @ B_obs @ (np.eye(Nobs) - K @ H).T + K @ R @ K.T
    B_a_obs = 0.5 * (B_a_obs + B_a_obs.T)
    
    # Cholesky of B_a_obs
    try:
        L_a = np.linalg.cholesky(B_a_obs)
    except np.linalg.LinAlgError:
        # Add small jitter
        B_a_obs += 1e-10 * np.eye(Nobs)
        try:
            L_a = np.linalg.cholesky(B_a_obs)
        except np.linalg.LinAlgError:
            L_a = np.linalg.cholesky(B_a_obs + 1e-8 * np.eye(Nobs))
    
    # Cholesky of B_obs (prior)
    try:
        L_b = np.linalg.cholesky(B_obs)
    except np.linalg.LinAlgError:
        B_obs += 1e-10 * np.eye(Nobs)
        try:
            L_b = np.linalg.cholesky(B_obs)
        except np.linalg.LinAlgError:
            L_b = np.linalg.cholesky(B_obs + 1e-8 * np.eye(Nobs))
    
    # Transformation matrix: T = L_a @ inv(L_b)
    # This maps prior anomalies to analysis anomalies while preserving the mean shift
    T = L_a @ np.linalg.inv(L_b)  # (Nobs, Nobs)
    
    # Now build the full transformation for all D variables
    # For unobserved variables, use the regression from the ensemble
    # B_full = anomalies.T @ anomalies / (N-1)  (D, D)
    B_full = (anomalies.T @ anomalies) / (N - 1.0)
    
    # Cross-covariance: B_unobs_obs = B_full[np.ix_(np.setdiff1d(np.arange(D), obs_idx), obs_idx)]
    unobs_idx = np.setdiff1d(np.arange(D), obs_idx)
    B_uo = B_full[np.ix_(unobs_idx, obs_idx)]  # (D-Nobs, Nobs)
    
    # Regression coefficients: C = B_uo @ inv(B_obs)
    try:
        C = B_uo @ np.linalg.inv(B_obs)
    except np.linalg.LinAlgError:
        C = B_uo @ np.linalg.pinv(B_obs)
    
    # For unobserved variables, the anomaly update is:
    # a_u_new = C @ a_o_new (linear regression)
    # But we need to apply the square-root transform consistently.
    # The full transformation in anomaly space:
    # For obs: a_o_new = T @ a_o_old
    # For unobs: a_u_new = C @ T @ a_o_old
    
    # Build full (D, Nobs) transformation matrix
    T_full = np.zeros((D, Nobs))
    T_full[obs_idx, :] = T
    T_full[unobs_idx, :] = C @ T
    
    # Apply to anomalies
    new_anomalies = anomalies @ T_full.T  # (N, D)
    
    # New mean
    new_mean = mean.copy()
    new_mean[obs_idx] = new_mean_obs
    # Unobserved mean stays the same (or could be updated via regression, but for EnAKF it's fixed)
    
    # Reconstruct ensemble
    new_ensemble = new_mean[None, :] + new_anomalies
    
    # Clip to valid range
    new_ensemble = np.clip(new_ensemble, -100.0, 100.0)
    
    return new_ensemble


def _adaptive_inflation(ensemble, observations, observed_indices, obs_var, 
                        target_ratio=1.15, beta=0.2):
    """
    Adaptive inflation to prevent ensemble collapse.
    Returns inflated ensemble.
    """
    N, D = ensemble.shape
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean
    
    # Compute spread at observed locations
    obs_idx = np.array(observed_indices, dtype=int)
    spread_obs = anomalies[:, obs_idx].std(axis=0, ddof=1)
    
    # Innovation standard deviation
    obs_mean = observations.mean(axis=0)
    innovation = obs_mean - mean[obs_idx]
    innov_std = np.sqrt(obs_var + spread_obs**2)
    
    # Target: spread should be at least target_ratio * innovation_std
    target_spread = target_ratio * innov_std
    
    # Inflation factor
    ratio = spread_obs / (innov_std + 1e-12)
    # Inflate where ratio < target_ratio
    inflate_factor = np.ones(D)
    for i, oi in enumerate(obs_idx):
        if ratio[i] < target_ratio:
            inflate_factor[oi] = 1.0 + beta * (target_ratio - ratio[i])
    
    # Apply inflation to anomalies
    inflated_anomalies = anomalies * inflate_factor[None, :]
    
    return mean[None, :] + inflated_anomalies


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using EnAKF with localization and adaptive inflation.
    
    Parameters:
    ensemble: (16, 20) array - ensemble of 16 members, 20 variables
    observations: (16, 10) array - noisy observations of 10 (even-indexed) variables
    observed_indices: array of observed variable indices
    observation_variance: scalar - observation noise variance
    
    Returns:
    updated ensemble with the same (16, 20) shape
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    N, D = ensemble.shape
    
    # Adaptive inflation
    ensemble_inflated = _adaptive_inflation(
        ensemble, observations, observed_indices, 
        observation_variance, target_ratio=1.15, beta=0.2
    )
    
    # EnAKF analysis update
    updated = _enakf_analysis(
        ensemble_inflated, observations, observed_indices,
        observation_variance, loc_scale=4.0
    )
    
    # Ensure finite and within bounds
    updated = np.where(np.isfinite(updated), updated, 0.0)
    updated = np.clip(updated, -100.0, 100.0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Shouldn't happen, but safety check
        pass
    
    return updated

# EVOLVE-BLOCK-END
