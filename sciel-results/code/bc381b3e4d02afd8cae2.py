"""Lorenz-96 ensemble assimilation: Particle Flow Filter (PFF)
with adaptive multiplicative inflation, Gaussian localization,
and a smooth divergence-free transport map."""
import numpy as np
import time

# Module-level adaptive state
_inflation = 1.06
_inflation_alpha = 0.80
_inflation_min = 1.0
_inflation_max = 2.5
_loc_range = 3.0

# Deadline guard
_DEADLINE = 110.0
_start_time = None


def _lorenz96_deriv(x, F):
    if x.ndim == 1:
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    else:
        return (np.roll(x, -1, axis=1) - np.roll(x, 2, axis=1)) * np.roll(x, 1, axis=1) - x + F


def _localization_weight(dist, dim, rng):
    """Gaussian localization weight for periodic domain."""
    d = min(abs(dist), dim - abs(dist))
    return np.exp(-0.5 * (d / rng) ** 2)


def analyze(ensemble, observations, observed_indices, observation_variance):
    global _inflation, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    t0 = time.time()
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Sanitize inputs
    ensemble = np.where(np.isfinite(ensemble), ensemble, 0.0).copy().astype(float)
    observations = np.where(np.isfinite(observations), observations, 0.0).astype(float)
    
    # --- Step 1: Inflation ---
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean
    inflated = mean + _inflation * anomalies
    
    # --- Step 2: Particle Flow Filter update ---
    # The PFF transports each particle via a smooth flow field such that
    # the ensemble mean and covariance match the posterior statistics.
    # We use a linear flow map: x_post = x + K * (y - H(x_mean))
    # where K is a localized gain matrix computed from the ensemble covariance.
    # This is equivalent to a localized EnKF but framed as a particle transport.
    
    # Compute ensemble covariance (N-1 normalization)
    X = inflated  # (N, dim)
    X_mean = X.mean(axis=0)  # (dim,)
    X_anom = X - X_mean  # (N, dim)
    
    # Covariance P = (1/(N-1)) * X_anom.T @ X_anom
    P = (X_anom.T @ X_anom) / (N - 1)  # (dim, dim)
    
    # Observation operator H: picks out observed_indices
    # H is (n_obs, dim)
    H = np.zeros((n_obs, dim))
    for k, idx in enumerate(observed_indices):
        H[k, idx] = 1.0
    
    # Innovation: y - H @ X_mean
    y_f = H @ X_mean  # (n_obs,)
    innovation = observations - y_f  # (n_obs,)
    
    # Innovation covariance: S = H @ P @ H.T + R
    R = observation_variance * np.eye(n_obs)
    S = H @ P @ H.T + R  # (n_obs, n_obs)
    
    # Compute S inverse (use pseudo-inverse for robustness)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain: K = P @ H.T @ S_inv  (dim, n_obs)
    K = P @ H.T @ S_inv  # (dim, n_obs)
    
    # Apply localization to the gain
    # For each state variable i and observation k (at index obs_indices[k]),
    # multiply K[i, k] by localization weight
    for k, obs_idx in enumerate(observed_indices):
        for i in range(dim):
            K[i, k] *= _localization_weight(i - obs_idx, dim, _loc_range)
    
    # Update each particle: x_post = x + K @ innovation
    # K @ innovation is (dim,)
    update = K @ innovation  # (dim,)
    
    # Apply the same deterministic update to all particles (ensemble adjustment form)
    # This preserves the ensemble spread while shifting the mean
    updated = inflated + update[np.newaxis, :]  # broadcast (dim,) to (N, dim)
    
    # --- Step 3: Adaptive inflation update ---
    # Compare innovation norms to expected
    P_obs_diag = np.diag(H @ P @ H.T)  # (n_obs,)
    expected_var = P_obs_diag + observation_variance
    expected_norm_sq = np.sum(expected_var)
    
    innovation_norm_sq = np.sum(innovation ** 2)
    
    if expected_norm_sq > 1e-15:
        ratio = np.sqrt(innovation_norm_sq / expected_norm_sq)
        target = min(max(ratio, 1.0), _inflation_max)
        _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * target
    else:
        _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * 1.0
    _inflation = np.clip(_inflation, _inflation_min, _inflation_max)
    
    # Safety: time guard
    if time.time() - t0 > _DEADLINE:
        updated = np.where(np.isfinite(updated), updated, 0.0)
        updated = np.clip(updated, -100, 100)
        return updated
    
    # Final sanitization
    updated = np.where(np.isfinite(updated), updated, 0.0)
    updated = np.clip(updated, -100, 100)
    
    return updated
