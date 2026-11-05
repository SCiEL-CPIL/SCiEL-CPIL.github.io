"""Lorenz-96 ensemble assimilation: localized POEnKF with spatially adaptive inflation."""
import numpy as np
import time

# Module-level adaptive state
# Vector of inflation factors for each state variable
_inflation_vec = np.ones(20)
_inflation_alpha = 0.70
_inflation_min = 1.0
_inflation_max = 2.5
_loc_range = 2.5  # Slightly tighter localization might help

# Cache for localization weights
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
    global _inflation_vec
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
    
    # --- Spatially Adaptive Inflation ---
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean  # (N, dim)
    
    # Apply spatially adaptive inflation
    # inflated[i, j] = mean[j] + _inflation_vec[j] * anomalies[i, j]
    inflated = mean[np.newaxis, :] + _inflation_vec[np.newaxis, :] * anomalies
    
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
    
    # Apply localization to K
    # K_loc[s, j] = K[s, j] * exp(-0.5 * (d(s, obs_j) / loc_range)^2)
    K_loc = K.copy()
    for j, obs_idx in enumerate(observed_indices):
        # Use cached weights if possible, but the weight depends on state index s
        # So we compute for each state s
        for s in range(dim):
            d = min(abs(s - obs_idx), dim - abs(s - obs_idx))
            K_loc[s, j] *= np.exp(-0.5 * (d / _loc_range) ** 2)
    
    # Update each member (Perturbed Observation EnKF)
    innovations_all = observations[:, np.newaxis] - y_f  # (n_obs, N)
    updates = K_loc @ innovations_all  # (dim, N)
    updated = inflated + updates.T  # (N, dim)
    
    # --- Update Spatially Adaptive Inflation ---
    # For each state variable j, compute a localized innovation metric
    # We use the localization weights from each observation to project innovations to state j
    # I_j = sum_k w(j, obs_k) * innovation_k
    # E_j = sum_{k,l} w(j, obs_k) * S_kl * w(j, obs_l)
    
    new_inflation = np.ones(dim)
    for j in range(dim):
        # Vector of weights from all observations to state j
        w_vec = np.zeros(n_obs)
        for k, obs_idx in enumerate(observed_indices):
            d = min(abs(j - obs_idx), dim - abs(j - obs_idx))
            w_vec[k] = np.exp(-0.5 * (d / _loc_range) ** 2)
        
        # Localized innovation
        I_j = np.dot(w_vec, innovation)
        
        # Localized expected variance
        E_j = w_vec @ S @ w_vec
        
        if E_j > 1e-12:
            ratio = abs(I_j) / np.sqrt(E_j)
            # Target inflation: if ratio > 1, we need more spread.
            # Use a soft target to avoid instability
            target = min(max(ratio, 1.0), _inflation_max)
            # Update with exponential moving average
            new_inflation[j] = _inflation_alpha * _inflation_vec[j] + (1 - _inflation_alpha) * target
        else:
            # If no signal, decay to 1.0
            new_inflation[j] = _inflation_alpha * _inflation_vec[j] + (1 - _inflation_alpha) * 1.0
            
    _inflation_vec = np.clip(new_inflation, _inflation_min, _inflation_max)
    
    # Safety: time guard
    if time.time() - t0 > 100:
        # Return something valid quickly
        return np.clip(np.where(np.isfinite(updated), updated, mean), -100, 100)
    
    # Final sanitization
    updated = np.where(np.isfinite(updated), updated, 0.0)
    updated = np.clip(updated, -100, 100)
    
    return updated
