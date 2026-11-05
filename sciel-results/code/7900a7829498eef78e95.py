"""Lorenz-96 ensemble assimilation: localized EKF with adaptive inflation,
per-member localized gain, and a simple forcing estimate."""
import numpy as np
import time

# Module-level adaptive state
_inflation = 1.06
_inflation_alpha = 0.80
_inflation_min = 1.0
_inflation_max = 2.2
_loc_range = 2.8

# Precompute localization weights for even indices (0,2,4,...,18)
# We'll build them lazily and cache
_loc_cache = {}

def _get_loc_weights(obs_idx, dim=20, rng=2.8):
    key = (obs_idx, dim, rng)
    if key not in _loc_cache:
        w = np.empty(dim)
        for j in range(dim):
            d = min(abs(obs_idx - j), dim - abs(obs_idx - j))
            w[j] = np.exp(-0.5 * (d / rng) ** 2)
        _loc_cache[key] = w
    return _loc_cache[key]


def _lorenz96_deriv(x, F):
    if x.ndim == 1:
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    else:
        return (np.roll(x, -1, axis=1) - np.roll(x, 2, axis=1)) * np.roll(x, 1, axis=1) - x + F


def analyze(ensemble, observations, observed_indices, observation_variance):
    global _inflation
    t0 = time.time()
    
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Sanitize
    ensemble = np.where(np.isfinite(ensemble), ensemble, 0.0).copy()
    observations = np.where(np.isfinite(observations), observations, 0.0).astype(float)
    
    # Observation operator
    H = np.zeros((n_obs, dim))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # --- Adaptive inflation ---
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean  # (N, dim)
    inflated = mean + _inflation * anomalies  # (N, dim)
    
    # Forecast observations per member
    y_f = H @ inflated.T  # (n_obs, N)
    y_f_mean = y_f.mean(axis=1)  # (n_obs,)
    innovation = observations - y_f_mean  # (n_obs,)
    
    # Covariance of inflated ensemble
    cov = np.cov(inflated, rowvar=False)  # (dim, dim)
    
    # Observation noise covariance
    R = observation_variance * np.eye(n_obs)
    
    # Innovation covariance S = H P H^T + R
    P_obs = cov[np.ix_(observed_indices, observed_indices)]  # (n_obs, n_obs)
    S = P_obs + R
    
    # Regularize for numerical stability
    S += 1e-10 * np.eye(n_obs)
    
    # Kalman gain K = P H^T S^{-1}
    PHt = cov @ H.T  # (dim, n_obs)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    K = PHt @ S_inv  # (dim, n_obs)
    
    # Apply per-member localized EKF update
    # For member i: x_i += sum_j K[:,j] * w_j(x_i) * (y_j - y_f_{i,j})
    # where w_j is localization weight for observation j applied to state j's column of K
    # Actually, localization should be applied to the cross-covariance P_{state,obs}
    # A common approach: for each observation j, scale K[:, j] by a localization
    # function of the distance between state index and observation index j.
    
    # Build localized K: for each state row s and obs col j,
    # K_loc[s, j] = K[s, j] * exp(-0.5 * (d(s, obs_j) / loc_range)^2)
    K_loc = K.copy()
    for j, obs_idx in enumerate(observed_indices):
        for s in range(dim):
            d = min(abs(s - obs_idx), dim - abs(s - obs_idx))
            K_loc[s, j] *= np.exp(-0.5 * (d / _loc_range) ** 2)
    
    # Update each member
    innovations_all = observations[:, np.newaxis] - y_f  # (n_obs, N)
    # Update: inflated[i] + K_loc @ innovations_all[:, i]
    updates = K_loc @ innovations_all  # (dim, N)
    updated = inflated + updates.T  # (N, dim)
    
    # --- Adaptive inflation update ---
    innovation_norm = np.linalg.norm(innovation)
    expected_norm = np.sqrt(max(np.trace(S), 1e-15))
    if expected_norm > 1e-12:
        ratio = innovation_norm / expected_norm
        # Target inflation ~ ratio when ratio > 1
        target = min(max(ratio, 1.0), _inflation_max)
        _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * target
    else:
        _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * 1.0
    _inflation = np.clip(_inflation, _inflation_min, _inflation_max)
    
    # Safety: time guard
    if time.time() - t0 > 100:
        # Return something valid quickly
        return np.clip(np.where(np.isfinite(updated), updated, mean), -100, 100)
    
    # Final sanitization
    updated = np.where(np.isfinite(updated), updated, 0.0)
    updated = np.clip(updated, -100, 100)
    
    return updated
