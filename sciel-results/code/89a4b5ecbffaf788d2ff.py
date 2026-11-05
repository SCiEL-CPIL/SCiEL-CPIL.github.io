"""Lorenz-96 ensemble assimilation: localized EKF with adaptive inflation,
spatially-weighted gain, and ensemble-mean bias correction."""
import numpy as np
import time

# Module-level adaptive state
_inflation = 1.04
_inflation_alpha = 0.82
_inflation_min = 1.0
_inflation_max = 1.8
_loc_range = 2.5

# Precompute localization weights for even indices (0,2,4,...,18)
_loc_cache = {}

def _get_loc_weights(obs_idx, dim=20, rng=2.5):
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
    
    # Apply per-member localized EKF update with spatial weighting
    # Build localized K: for each state row s and obs col j,
    # K_loc[s, j] = K[s, j] * exp(-0.5 * (d(s, obs_j) / loc_range)^2)
    K_loc = K.copy()
    for j, obs_idx in enumerate(observed_indices):
        w = _get_loc_weights(obs_idx, dim, _loc_range)
        K_loc[:, j] *= w
    
    # Update each member
    innovations_all = observations[:, np.newaxis] - y_f  # (n_obs, N)
    updates = K_loc @ innovations_all  # (dim, N)
    updated = inflated + updates.T  # (N, dim)
    
    # --- Ensemble-mean bias correction ---
    # The ensemble mean may have a systematic offset from the observations.
    # Apply a gentle correction to pull the mean toward the observed values.
    # This is a form of "mean bias correction" that helps when the ensemble
    # is centered slightly off the true state.
    mean_after = updated.mean(axis=0)
    # Compute the mean innovation in state space (projected through H)
    mean_obs_innov = innovation  # (n_obs,)
    # Apply a small correction proportional to the innovation
    # Scale by 0.3 to be gentle
    bias_correction = np.zeros(dim)
    for i, idx in enumerate(observed_indices):
        bias_correction[idx] += 0.3 * mean_obs_innov[i]
    # Spread the correction to nearby states using localization
    for i, idx in enumerate(observed_indices):
        w = _get_loc_weights(idx, dim, _loc_range)
        bias_correction += 0.1 * mean_obs_innov[i] * w
    updated += bias_correction[np.newaxis, :]
    
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
        return np.clip(np.where(np.isfinite(updated), updated, mean), -100, 100)
    
    # Final sanitization
    updated = np.where(np.isfinite(updated), updated, 0.0)
    updated = np.clip(updated, -100, 100)
    
    return updated
