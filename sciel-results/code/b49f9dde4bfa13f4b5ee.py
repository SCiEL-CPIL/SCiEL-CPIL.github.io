"""Ensemble Kalman Filter with localization and adaptive inflation for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Module-level state for adaptive inflation
_inflation_factor = 1.0
_prev_ensemble = None
_prev_observed_indices = None

def _lorenz96_rhs(x, F):
    """Lorenz-96 right-hand side."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _localize_cov(cov, n, decay=5.0):
    """Apply exponential localization to covariance matrix."""
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])
    # Handle periodicity: distance is min(dist, n-dist)
    dist = np.minimum(dist, n - dist)
    weight = np.exp(-dist / decay)
    return cov * weight

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis update with localization and adaptive inflation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    global _inflation_factor, _prev_ensemble, _prev_observed_indices
    
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin before 120s limit
    
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Compute ensemble mean and anomaly covariance
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean[None, :]
    
    # Sample covariance (with localization on observed components)
    # Full covariance
    cov_full = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Extract observed-observed and observed-unobserved blocks
    obs_idx = observed_indices
    unobs_idx = np.setdiff1d(np.arange(n_state), obs_idx)
    
    cov_oo = cov_full[np.ix_(obs_idx, obs_idx)]
    cov_ou = cov_full[np.ix_(obs_idx, unobs_idx)]
    cov_uo = cov_full[np.ix_(unobs_idx, obs_idx)]
    cov_uu = cov_full[np.ix_(unobs_idx, unobs_idx)]
    
    # Apply localization to the observed-observed covariance
    cov_oo_loc = _localize_cov(cov_oo, n_obs, decay=3.0)
    
    # Observation covariance
    R = observation_variance * np.eye(n_obs)
    
    # Kalman gain computation
    # S = H P H^T + R
    S = cov_oo_loc + R
    S_inv = np.linalg.inv(S)
    
    # K = P H^T S^{-1}
    K = cov_ou @ S_inv  # (n_unobs, n_obs)
    K_obs = cov_oo_loc @ S_inv  # (n_obs, n_obs)
    
    # Innovation
    innovation = observations - mean[obs_idx]
    
    # Update mean
    mean_new = mean.copy()
    mean_new[obs_idx] += K_obs @ innovation
    mean_new[unobs_idx] += K @ innovation
    
    # Update anomalies (to preserve ensemble spread)
    # Use the square root form or direct update
    # Direct update: a' = (I - K H) a
    # H maps unobserved to observed: H_obs is identity on obs_idx
    # For observed: a_obs' = (I - K_obs) a_obs
    # For unobserved: a_unobs' = (I - K H) a_unobs, but H only acts on obs part
    # Actually, the full update is:
    # a' = a - K H a
    # where H a picks out the observed components
    
    # Let's do it properly with the full gain matrix
    # Full gain: K_full has shape (n_state, n_obs)
    K_full = np.zeros((n_state, n_obs))
    K_full[obs_idx, :] = K_obs
    K_full[unobs_idx, :] = K
    
    # H_full: (n_obs, n_state)
    H_full = np.zeros((n_obs, n_state))
    H_full[np.arange(n_obs), obs_idx] = 1.0
    
    # Update each ensemble member
    result = np.empty_like(ensemble)
    for i in range(n_members):
        a = anomalies[i]
        # a' = a - K_full @ (H_full @ a)
        Ha = H_full @ a  # (n_obs,)
        a_new = a - K_full @ Ha
        result[i] = mean_new + a_new
    
    # Adaptive inflation
    # Compute ensemble spread at observed locations
    spread_obs = np.std(result[:, obs_idx], axis=0)
    obs_spread_mean = np.mean(spread_obs)
    
    # Compare to observation noise std
    obs_std = np.sqrt(observation_variance)
    
    # If spread is too small relative to observation noise, inflate
    # Target: spread should be at least comparable to observation noise
    if obs_spread_mean < 0.5 * obs_std:
        _inflation_factor = min(_inflation_factor * 1.05, 1.5)
    elif obs_spread_mean > 2.0 * obs_std:
        _inflation_factor = max(_inflation_factor * 0.98, 1.0)
    
    # Apply inflation around the mean
    if _inflation_factor > 1.0:
        for i in range(n_members):
            result[i] = mean_new + _inflation_factor * (result[i] - mean_new)
    
    # Clamp to valid range
    result = np.clip(result, -100.0, 100.0)
    
    # Ensure finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Update global state
    _prev_ensemble = result.copy()
    _prev_observed_indices = observed_indices.copy()
    
    return result

# EVOLVE-BLOCK-END
