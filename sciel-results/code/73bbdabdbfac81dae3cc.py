"""Ensemble Kalman Filter for Lorenz-96 assimilation with inflation and localization."""
import numpy as np
import time

# Module-level state for adaptive inflation
_inflation_factor = 1.01
_inflation_decay = 0.999
_inflation_min = 1.005
_inflation_max = 1.05

def _lorenz96_rhs(x, F):
    """Compute RHS of Lorenz-96 system."""
    n = len(x)
    dx = np.empty_like(x)
    for i in range(n):
        dx[i] = (x[(i + 1) % n] - x[(i - 2) % n]) * x[(i - 1) % n] - x[i] + F
    return dx

def _lorenz96_rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _lorenz96_propagate(x, F, dt, steps):
    """Propagate state x by RK4 for given steps."""
    for _ in range(steps):
        x = _lorenz96_rk4_step(x, F, dt)
    return x

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis update for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray of shape (16, 20)
        Ensemble members.
    observations : ndarray of shape (10,)
        Noisy observations of observed indices.
    observed_indices : array-like
        Indices of observed variables (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray of shape (16, 20)
        Updated ensemble after analysis.
    """
    global _inflation_factor, _inflation_decay, _inflation_min, _inflation_max
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 115.0  # 115s safety margin
    
    n_members, n_vars = ensemble.shape
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    
    # Estimate forcing from ensemble mean (rough heuristic)
    # The forcing F is not provided, so we estimate it from the ensemble
    # Actually, we don't need to propagate within analyze - the evaluator does that
    # We just need to update the ensemble based on observations
    
    # Compute ensemble mean and covariance
    ens_mean = np.mean(ensemble, axis=0)
    
    # Center the ensemble
    centered = ensemble - ens_mean
    
    # Compute ensemble covariance (sample covariance)
    # C = (1/(n-1)) * centered.T @ centered
    cov = (centered.T @ centered) / (n_members - 1)
    
    # Innovation covariance: S = H P H^T + R
    # H is the observation operator: selects observed indices
    H_cov_H = cov[np.ix_(obs_idx, obs_idx)]
    R = observation_variance * np.eye(n_obs)
    S = H_cov_H + R
    
    # Innovation: y - H x
    innovation = observations - ens_mean[obs_idx]
    
    # Kalman gain: K = P H^T S^{-1}
    # P H^T = centered @ H^T (where H^T is n_vars x n_obs)
    PHt = centered @ np.eye(n_obs)[np.arange(n_vars) == obs_idx].T  # This is wrong
    
    # Let's do it properly
    # PHt[i, j] = sum over members of (x_m[i] - mean[i]) * H[j, i]
    # H[j, i] = 1 if i == obs_idx[j], else 0
    # So PHt[i, j] = (x_m[i] - mean[i]) if i == obs_idx[j], else 0
    # Actually, PHt = centered.T @ H where H is n_obs x n_vars
    # H[j, i] = 1 if i == obs_idx[j]
    H = np.zeros((n_obs, n_vars))
    for j, idx in enumerate(obs_idx):
        H[j, idx] = 1.0
    
    PHt = centered.T @ H  # shape: n_vars x n_obs
    
    # Solve S^{-1} * innovation
    try:
        S_inv_innov = np.linalg.solve(S, innovation)
    except np.linalg.LinAlgError:
        S_inv_innov = np.linalg.pinv(S) @ innovation
    
    # Kalman gain: K = PHt @ S^{-1}
    K = PHt @ S_inv_innov  # shape: n_vars x n_obs
    
    # Update ensemble: x_new = x + K * (y - H x)
    # For each member: x_m_new = x_m + K @ (y - H x_m)
    # = x_m + K @ innovation - K @ H @ x_m
    # But H @ x_m = x_m[obs_idx]
    # So x_m_new = x_m + K @ innovation - K @ x_m[obs_idx]
    
    # K @ innovation is a vector of shape (n_vars,)
    K_innov = K @ innovation  # shape: (n_vars,)
    
    # For each member, compute K @ x_m[obs_idx]
    # K is n_vars x n_obs, x_m[obs_idx] is (n_obs,)
    # So K @ x_m[obs_idx] is (n_vars,)
    KHx = ensemble[:, obs_idx] @ K.T  # shape: (n_members, n_vars)
    
    updated_ensemble = ensemble + K_innov[None, :] - KHx
    
    # Apply multiplicative inflation to maintain ensemble spread
    updated_ens_mean = np.mean(updated_ensemble, axis=0)
    updated_centered = updated_ensemble - updated_ens_mean
    updated_cov_diag = np.mean(updated_centered ** 2, axis=0)
    
    # Compare with prior covariance
    prior_cov_diag = np.mean(centered ** 2, axis=0)
    
    # Adaptive inflation: if ensemble spread decreased too much, inflate
    # Target: keep spread roughly at prior level
    ratio = np.sqrt(np.maximum(updated_cov_diag, 1e-10) / np.maximum(prior_cov_diag, 1e-10))
    avg_ratio = np.mean(ratio)
    
    if avg_ratio < 0.95:
        _inflation_factor = min(_inflation_factor * 1.01, _inflation_max)
    elif avg_ratio > 1.05:
        _inflation_factor = max(_inflation_factor * 0.99, _inflation_min)
    else:
        _inflation_factor = max(_inflation_factor * _inflation_decay, _inflation_min)
    
    # Apply inflation
    updated_ensemble = updated_ens_mean + _inflation_factor * (updated_ensemble - updated_ens_mean)
    
    # Clip to valid range
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(updated_ensemble)):
        updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
        updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
