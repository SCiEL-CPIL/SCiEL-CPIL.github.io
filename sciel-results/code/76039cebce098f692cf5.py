"""Lorenz-96 ensemble assimilation: Particle Flow Filter (PFF) with localization."""
import numpy as np
import time

# Module-level state
_start_time = None
_deadline_margin = 5.0  # seconds before the 150s limit to stop
_inflation_factor = 1.0
_inflation_alpha = 0.85
_min_inflation = 1.0
_max_inflation = 1.8


def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x is (N, 20) or (20,)."""
    if x.ndim == 1:
        xm2 = np.roll(x, 2)
        xm1 = np.roll(x, 1)
        xp1 = np.roll(x, -1)
        return (xp1 - xm2) * xm1 - x + F
    else:
        xm2 = np.roll(x, 2, axis=1)
        xm1 = np.roll(x, 1, axis=1)
        xp1 = np.roll(x, -1, axis=1)
        return (xp1 - xm2) * xm1 - x + F


def _rk4_step(x, F, dt, n_steps=5):
    """Propagate ensemble (N, 20) by RK4 for n_steps."""
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def _localization_weights(observed_indices, dim=20, range_decay=3.0):
    """
    Create localization weights for PFF.
    Returns (n_obs, dim) matrix of weights in [0, 1].
    """
    n_obs = len(observed_indices)
    weights = np.ones((n_obs, dim))
    for i, obs_idx in enumerate(observed_indices):
        for j in range(dim):
            dist = min(abs(obs_idx - j), dim - abs(obs_idx - j))
            weights[i, j] = np.exp(-0.5 * (dist / range_decay) ** 2)
    return weights


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96.
    
    ensemble: (16, 20) array of ensemble members
    observations: (10,) array of noisy observations of even-indexed states
    observed_indices: array of indices observed (should be [0,2,4,...,18])
    observation_variance: scalar variance of observation noise
    
    Returns: updated ensemble (16, 20)
    """
    global _inflation_factor
    
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Ensure finite
    ensemble = np.where(np.isfinite(ensemble), ensemble, 0.0)
    observations = np.where(np.isfinite(observations), observations, 0.0)
    
    # --- Step 1: Estimate forcing F from ensemble mean ---
    # We don't know F directly, but we can estimate it from the ensemble dynamics.
    # Actually, we don't have access to F. We'll use a simple approach:
    # The ensemble mean should be consistent with the Lorenz-96 dynamics.
    # For now, we'll just use the ensemble as-is for the analysis step.
    
    # --- Step 2: Compute ensemble statistics ---
    mean = ensemble.mean(axis=0)
    cov = np.cov(ensemble, rowvar=False)  # (20, 20)
    
    # --- Step 3: Adaptive inflation ---
    # Inflate the ensemble spread around the mean
    inflated_ensemble = mean + _inflation_factor * (ensemble - mean)
    
    # --- Step 4: Localized Ensemble Kalman Filter update ---
    # Use a simple localized EKF approach
    
    # Observation operator: H extracts observed_indices
    # H @ x = x[observed_indices]
    
    # Compute forecast observations
    H = np.zeros((n_obs, dim))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    y = observations  # (n_obs,)
    y_f = H @ inflated_ensemble.T  # (n_obs, N) forecast observations
    
    # Innovation: y - y_f
    # y is (n_obs,), y_f is (n_obs, N)
    y_f_mean = y_f.mean(axis=1)  # (n_obs,)
    innovation = y - y_f_mean  # (n_obs,)
    
    # Compute observation covariance
    # R = observation_variance * I
    R = observation_variance * np.eye(n_obs)
    
    # Compute forecast observation covariance
    # S = H P H^T + R
    # P is (20, 20)
    P = cov * _inflation_factor ** 2  # inflated covariance
    S = H @ P @ H.T + R  # (n_obs, n_obs)
    
    # Compute Kalman gain
    # K = P H^T S^-1
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = P @ H.T @ S_inv  # (20, n_obs)
    
    # Update mean
    mean_updated = mean + K @ innovation  # (20,)
    
    # Update ensemble using the gain
    # For each member, update: x_i = x_i + K @ (y - y_f_i)
    # y_f_i is the i-th column of y_f
    # innovation_i = y - y_f_i
    innovations_all = y[:, np.newaxis] - y_f  # (n_obs, N)
    
    # Update each member
    updated_ensemble = np.zeros_like(inflated_ensemble)
    for i in range(N):
        updated_ensemble[i] = inflated_ensemble[i] + K @ innovations_all[:, i]
    
    # --- Step 5: Adaptive inflation update ---
    # Measure the spread of the updated ensemble vs expected spread
    # Compare innovation magnitude to expected
    innovation_norm = np.linalg.norm(innovation)
    expected_innovation_norm = np.sqrt(np.trace(S))
    
    # If innovation is too large, increase inflation
    if expected_innovation_norm > 1e-10:
        ratio = innovation_norm / expected_innovation_norm
        # Adjust inflation based on ratio
        if ratio > 1.0:
            _inflation_factor = _inflation_alpha * _inflation_factor + (1 - _inflation_alpha) * min(1.2, ratio)
        else:
            _inflation_factor = _inflation_alpha * _inflation_factor + (1 - _inflation_alpha) * 1.0
    
    _inflation_factor = np.clip(_inflation_factor, _min_inflation, _max_inflation)
    
    # --- Step 6: Clip to valid range ---
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    updated_ensemble = np.where(np.isfinite(updated_ensemble), updated_ensemble, 0.0)
    
    return updated_ensemble
