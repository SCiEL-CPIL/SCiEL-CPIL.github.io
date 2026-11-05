"""Ensemble Kalman Filter for Lorenz-96 assimilation with localization and inflation."""
import numpy as np
import time

# Module-level state for adaptive inflation
_inflation_factor = 1.0
_inflation_min = 1.0
_inflation_max = 2.0

def _lorenz96_rhs(x, F):
    """Compute the Lorenz-96 right-hand side."""
    n = len(x)
    # Periodic indices
    x_im2 = np.roll(x, 2)
    x_im1 = np.roll(x, 1)
    x_ip1 = np.roll(x, -1)
    return (x_ip1 - x_im2) * x_im1 - x + F

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _localize_covariance(cov, obs_idx, state_dim, decay_scale=2.0):
    """Apply exponential localization to the covariance matrix."""
    n_obs = len(obs_idx)
    n_state = state_dim
    # Build distance matrix between observed and state variables
    # dist[i, j] = circular distance between obs_idx[i] and state variable j
    state_vars = np.arange(state_dim)
    dist = np.abs(np.subtract.outer(obs_idx, state_vars))
    dist = np.minimum(dist, state_dim - dist)  # circular distance
    
    # Exponential decay
    decay = np.exp(-dist / decay_scale)
    # Apply to covariance: cov[i, j] *= decay[i, j]
    # cov shape: (n_obs, n_state)
    return cov * decay

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis step with localization and adaptive inflation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,) or (16, 10)
        Noisy observations of the observed coordinates.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
    """
    global _inflation_factor
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    ens = np.asarray(ensemble, dtype=np.float64).copy()
    n_ens, n_state = ens.shape
    obs_idx = np.asarray(observed_indices, dtype=np.int64)
    n_obs = len(obs_idx)
    
    # Handle observation shape: could be (10,) or (16, 10)
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim == 2:
        obs_mean = np.mean(obs, axis=0)
    else:
        obs_mean = obs
    
    # Compute ensemble mean and covariance
    ens_mean = np.mean(ens, axis=0)
    
    # Center the ensemble
    ens_centered = ens - ens_mean
    ens_centered[obs_idx]  # just to ensure we're indexing correctly
    
    # Compute background covariance (16x20)
    cov_bg = (ens_centered.T @ ens_centered) / (n_ens - 1)
    
    # Localize the background covariance for observed variables
    # We need cov between observed and all state variables
    cov_obs_state = cov_bg[np.ix_(obs_idx, range(n_state))]
    cov_obs_state = _localize_covariance(cov_obs_state, obs_idx, n_state, decay_scale=2.5)
    
    # Observation covariance (diagonal)
    R = observation_variance * np.eye(n_obs)
    
    # Innovation: obs - H(x_bar) where H picks out observed components
    innovation = obs_mean - ens_mean[obs_idx]
    
    # Innovation covariance: S = H P H^T + R
    S = cov_obs_state[:, obs_idx] + R
    
    # Compute Kalman gain: K = P H^T S^{-1}
    # P H^T is cov_obs_state (n_obs x n_state)
    try:
        # Solve S^T X^T = (P H^T)^T => X = (P H^T) S^{-T}
        # K = P H^T S^{-1}, shape (n_obs, n_state) -> we need full K for state update
        # Actually K_full = P H^T S^{-1}, but we only have localized P H^T
        # Let's compute: K = cov_obs_state @ inv(S)
        # But cov_obs_state is (n_obs, n_state), S is (n_obs, n_obs)
        # K = cov_obs_state @ S^{-1}, shape (n_obs, n_state)
        S_inv = np.linalg.inv(S)
        K = cov_obs_state @ S_inv  # (n_obs, n_state)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        S_inv = np.linalg.pinv(S)
        K = cov_obs_state @ S_inv
    
    # Update each ensemble member
    # For each member: x_a = x_b + K (y - H x_b)
    # H x_b for each member: ens[:, obs_idx]
    update = np.zeros_like(ens)
    for i in range(n_ens):
        innov_i = obs_mean - ens[i, obs_idx]  # (n_obs,)
        update[i] = K.T @ innov_i  # (n_state,)
    
    ens_updated = ens + update
    
    # Adaptive inflation: check if spread is too small
    # Compute spread vs observation error
    spread_obs = np.std(ens_updated[:, obs_idx], axis=0)
    obs_err = np.abs(obs_mean - np.mean(ens_updated[:, obs_idx], axis=0))
    
    # If spread is too small relative to obs error, inflate
    ratio = np.mean(spread_obs) / (np.mean(obs_err) + 1e-10)
    if ratio < 0.8:
        _inflation_factor = min(_inflation_factor * 1.1, _inflation_max)
    elif ratio > 1.2:
        _inflation_factor = max(_inflation_factor * 0.95, _inflation_min)
    
    # Apply inflation around the mean
    if _inflation_factor > 1.001:
        ens_mean_updated = np.mean(ens_updated, axis=0)
        ens_updated = ens_mean_updated + _inflation_factor * (ens_updated - ens_mean_updated)
    
    # Ensure output is finite and within bounds
    ens_updated = np.clip(ens_updated, -100.0, 100.0)
    ens_updated = np.nan_to_num(ens_updated, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline check
    if time.time() > deadline:
        return ens_updated
    
    return ens_updated
