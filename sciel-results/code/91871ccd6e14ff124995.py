"""Ensemble Kalman Filter with localization and adaptive inflation for Lorenz-96."""
import numpy as np
import time

# Module-level state for adaptive inflation
_inflation_state = {'alpha': 1.0, 'min_alpha': 1.0, 'max_alpha': 2.0, 'decay': 0.95}

def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn 5th-order correlation function for localization."""
    d = np.abs(d)
    result = np.zeros_like(d)
    
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    dc = d[mask1] / c
    result[mask1] = (-15.0/4.0) * dc**5 + 10.0 * dc**4 - 5.0 * dc**3 + 1.0
    
    dc = d[mask2] / c
    result[mask2] = (1.0/12.0) * dc**5 - (5.0/4.0) * dc**4 + (5.0/2.0) * dc**3 - (5.0/4.0) * dc**2 + (25.0/12.0) * dc - 1.0/2.0
    
    dc = d[mask3] / c
    result[mask3] = (-1.0/48.0) * dc**5 + (1.0/4.0) * dc**4 - (15.0/8.0) * dc**3 + (5.0/4.0) * dc**2 - (25.0/12.0) * dc + 1.0/2.0
    
    return result

def _localize_cov(cov, n, c=1.5):
    """Apply Gaspari-Cohn localization to a covariance matrix."""
    indices = np.arange(n)
    dist = np.abs(indices[:, None] - indices[None, :])
    # For periodic systems, also consider wrap-around distance
    dist = np.minimum(dist, n - dist)
    weights = _gaspari_cohn(dist, c)
    return cov * weights

def _adaptive_inflation(ensemble, observations, observed_indices, observation_variance):
    """Adaptive multiplicative inflation based on ensemble spread vs innovation."""
    n_members = ensemble.shape[0]
    obs_count = len(observed_indices)
    
    # Compute ensemble mean and spread at observed locations
    ens_mean = ensemble[:, observed_indices].mean(axis=0)
    ens_spread = ensemble[:, observed_indices].std(axis=0, ddof=1)
    
    # Innovation: obs - ensemble mean
    innovation = observations - ens_mean
    innov_var = np.var(innovation) if obs_count > 1 else np.abs(innovation).mean()
    
    # Expected innovation variance from ensemble spread and observation error
    expected_innov_var = ens_spread**2 + observation_variance
    expected_innov_var = np.mean(expected_innov_var) if len(expected_innov_var) > 1 else expected_innov_var[0]
    
    # Ratio of actual to expected innovation variance
    if expected_innov_var > 1e-10:
        ratio = innov_var / expected_innov_var
    else:
        ratio = 1.0
    
    # Adjust inflation factor
    state = _inflation_state
    target = 1.0 + 0.5 * (ratio - 1.0)
    target = np.clip(target, state['min_alpha'], state['max_alpha'])
    
    # Smooth inflation adjustment
    state['alpha'] = state['decay'] * state['alpha'] + (1 - state['decay']) * target
    return state['alpha']

def analyze(ensemble, observations, observed_indices, observation_variance):
    """Ensemble Kalman Filter with localization and adaptive inflation."""
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    n_members, n_state = ensemble.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # Compute adaptive inflation
    alpha = _adaptive_inflation(ensemble, observations, obs_idx, observation_variance)
    
    # Apply multiplicative inflation about the ensemble mean
    ens_mean = ensemble.mean(axis=0)
    ensemble_inflated = ens_mean + alpha * (ensemble - ens_mean)
    
    # Compute ensemble covariance
    anomalies = ensemble_inflated - ens_mean
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Localize the covariance
    cov_localized = _localize_cov(cov, n_state, c=1.5)
    
    # Observation operator: select observed indices
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(obs_idx):
        H[i, idx] = 1.0
    
    # Observation covariance
    R = observation_variance * np.eye(n_obs)
    
    # Innovation
    innovation = observations - ens_mean[obs_idx]
    
    # Kalman gain computation
    # S = H P H^T + R
    H_cov = H @ cov_localized
    S = H_cov @ H.T + R
    
    # Solve for Kalman gain: K = P H^T S^{-1}
    try:
        # Use Cholesky for numerical stability
        L = np.linalg.cholesky(S)
        # Solve L Y^T = H_cov^T
        Y = np.linalg.solve(L, H_cov.T)
        # K = Y L^{-1}
        K = np.linalg.solve(L.T, Y).T
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        S_inv = np.linalg.pinv(S)
        K = cov_localized @ H.T @ S_inv
    
    # Apply Kalman update
    # Update all state variables
    ens_mean_updated = ens_mean + K @ innovation
    anomalies_updated = anomalies - K @ (H @ anomalies.T).T
    
    ensemble_updated = ens_mean_updated[None, :] + anomalies_updated
    
    # Clip to valid range
    ensemble_updated = np.clip(ensemble_updated, -100.0, 100.0)
    
    # Check for non-finite values
    if not np.all(np.isfinite(ensemble_updated)):
        ensemble_updated = np.nan_to_num(ensemble_updated, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ensemble_updated
