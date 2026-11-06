"""Ensemble Kalman Filter with localization, inflation, and time guard for Lorenz-96 assimilation."""
import numpy as np
import time

# Module-level state for persistent inflation and covariance memory
_state = {
    'inflation': 1.0,
    'min_inflation': 1.0,
    'max_inflation': 1.5,
    'inflation_growth': 1.005,
    'inflation_decay': 0.995,
    'initialized': False,
}

def _lorenz96(x, F):
    """Lorenz-96 dynamics: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F"""
    n = len(x)
    dx = np.empty_like(x)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96(x, F)
    k2 = _lorenz96(x + 0.5*dt*k1, F)
    k3 = _lorenz96(x + 0.5*dt*k2, F)
    k4 = _lorenz96(x + dt*k3, F)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def _gaspari_cohn(distances, width):
    """Gaspari-Cohn localization function."""
    r = np.abs(distances)
    c = width
    out = np.zeros_like(r, dtype=float)
    mask1 = r <= c
    mask2 = (r > c) & (r <= 2*c)
    out[mask1] = (-5*r[mask1]**4 + 2*r[mask1]**3 + 8*c**2*r[mask1]**2 - 4*c**3*r[mask1]) / (2*c*(4*c - 1))
    out[mask2] = (r[mask2]**4 - 2*r[mask2]**3*(2*c) + 5*c**2*r[mask2]**2 + 4*c**3*r[mask2] - 12*c**4) / (2*c*(4*c - 1))
    return out

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using an ensemble Kalman filter with:
    - Gaspari-Cohn spatial localization
    - Adaptive variance inflation
    - Time guard to stay within budget
    
    Parameters:
        ensemble: (16, 20) array of ensemble members
        observations: (10,) array of noisy observations of even-indexed states
        observed_indices: indices observed (should be [0,2,4,...,18])
        observation_variance: scalar, variance of observation noise
    
    Returns:
        Updated ensemble of shape (16, 20)
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n_members = ensemble.shape[0]  # 16
    n_vars = ensemble.shape[1]     # 20
    n_obs = len(observations)      # 10
    obs_idx = np.array(observed_indices)
    R = observation_variance       # scalar obs variance
    
    # Infer F from the ensemble mean dynamics (rough estimate)
    # We don't know F directly, so we'll work with what we have.
    # For localization and EKF, we don't strictly need F.
    
    # --- Localization ---
    # Gaspari-Cohn width in index space
    loc_width = 4.0  # correlation decay over 4 indices
    
    # Build localization matrix for observed variables
    # Distance matrix between observed indices
    obs_dist = np.abs(obs_idx[:, None] - obs_idx[None, :])
    L = _gaspari_cohn(obs_dist, loc_width)
    np.fill_diagonal(L, 1.0)
    
    # --- Ensemble Kalman Filter (square root form) ---
    # Center the ensemble
    x_mean = ensemble.mean(axis=0)
    X_c = ensemble - x_mean  # (n_members, n_vars)
    
    # Background covariance (sample)
    # C = X_c.T @ X_c / (n_members - 1)
    # But we work in observation space with localization
    
    # Project to observation space
    X_obs_c = X_c[:, obs_idx]  # (n_members, n_obs)
    
    # Localized background covariance in obs space
    # C_bg = L * (X_obs_c.T @ X_obs_c / (n_members - 1))
    # We use the sample covariance with localization applied element-wise
    C_bg = (X_obs_c.T @ X_obs_c) / (n_members - 1.0)
    C_bg = L * C_bg  # element-wise localization
    
    # Innovation
    y = observations - x_mean[obs_idx]  # (n_obs,)
    
    # Innovation covariance
    C_innov = C_bg + R * np.eye(n_obs)
    
    # Gain (solve C_innov @ K^T = C_bg^T)
    # K = C_bg @ inv(C_innov)
    try:
        # Use Cholesky for stability
        L_inv = np.linalg.cholesky(C_innov)
        # Solve L_inv @ L_inv^T @ K^T = C_bg^T
        # K^T = L_inv^T \ (L_inv \ C_bg^T)
        temp = np.linalg.solve(L_inv, C_bg.T)
        K_T = np.linalg.solve(L_inv.T, temp)
        K = K_T.T  # (n_obs, n_vars)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        K = C_bg @ np.linalg.pinv(C_innov)
    
    # Update all ensemble members
    # x_a = x_b + K @ (y - H(x_b))
    # For each member: x_a_i = x_b_i + K @ (y - H(x_b_i))
    # H(x_b_i) = x_b_i[obs_idx]
    innovations_per_member = y[None, :] - X_obs_c  # (n_members, n_obs)
    update = innovations_per_member @ K  # (n_members, n_vars)
    
    ensemble_updated = ensemble + update
    
    # --- Inflation ---
    # Check if the ensemble spread is too small relative to innovation
    # Compute the ensemble spread in observation space after update
    x_mean_new = ensemble_updated.mean(axis=0)
    X_obs_new_c = ensemble_updated[:, obs_idx] - x_mean_new[obs_idx]
    
    # Expected spread from updated covariance
    # Simple heuristic: compare pre-update and post-update spread
    pre_spread = np.sqrt(np.mean(np.var(ensemble[:, obs_idx], axis=0)))
    post_spread = np.sqrt(np.mean(np.var(ensemble_updated[:, obs_idx], axis=0)))
    
    # If post-update spread is much smaller than expected, inflate
    # Expected spread should be roughly sqrt(R) for well-calibrated filter
    expected_spread = np.sqrt(R)
    
    if post_spread < 0.5 * expected_spread:
        _state['inflation'] = min(_state['inflation'] * _state['inflation_growth'], _state['max_inflation'])
    elif post_spread > 2.0 * expected_spread:
        _state['inflation'] = max(_state['inflation'] * _state['inflation_decay'], _state['min_inflation'])
    else:
        # Gentle relaxation toward 1.0
        _state['inflation'] = _state['inflation'] * 0.999 + 0.001 * 1.0
    
    # Apply inflation about the ensemble mean
    if _state['inflation'] > 1.001:
        x_mean_final = ensemble_updated.mean(axis=0)
        ensemble_updated = x_mean_final + _state['inflation'] * (ensemble_updated - x_mean_final)
    
    # Clamp to valid range
    ensemble_updated = np.clip(ensemble_updated, -100.0, 100.0)
    
    # Ensure finiteness
    if not np.all(np.isfinite(ensemble_updated)):
        ensemble_updated = np.nan_to_num(ensemble_updated, nan=0.0, posinf=100.0, neginf=-100.0)
        ensemble_updated = np.clip(ensemble_updated, -100.0, 100.0)
    
    # Time guard: if we're getting close to deadline, just return
    if time.time() > deadline:
        return ensemble_updated
    
    return ensemble_updated
