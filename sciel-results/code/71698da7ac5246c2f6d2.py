import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) update with localization and adaptive inflation.
    
    Parameters:
    ensemble: (M, N) array of ensemble states, M=16, N=20
    observations: (M, K) array where K is number of observed indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    Updated ensemble with same (M, N) shape
    """
    start_time = time.time()
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure inputs are numpy arrays
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    
    # Result starts as a copy of the ensemble
    result = ensemble.copy()
    
    # Compute ensemble mean
    mean = np.mean(ensemble, axis=0)
    
    # Compute anomaly matrix (M, N)
    anomalies = ensemble - mean
    
    # Covariance: (N, N)
    cov = (anomalies.T @ anomalies) / (M - 1)
    
    # Localization: use a broader scale for better long-range correlation handling
    localization_scale = 5.0
    indices = np.arange(N)
    dist = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    dist = np.minimum(dist, N - dist)
    
    # Use a smoother localization function (quintic)
    loc_factor = np.ones_like(dist)
    for i in range(N):
        for j in range(N):
            d = dist[i, j]
            if d <= 1.0:
                loc_factor[i, j] = 1.0
            elif d <= 3.0:
                # Quintic polynomial that goes from 1 at d=1 to 0 at d=3
                x = (d - 1.0) / 2.0
                loc_factor[i, j] = 1.0 - 10.0 * x**5 + 15.0 * x**4 - 6.0 * x**3
            else:
                loc_factor[i, j] = 0.0
    
    # Localize the covariance
    cov_loc = cov * loc_factor
    
    # Observation operator: H is (K, N)
    H = np.zeros((K, N))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation covariance R
    R = observation_variance * np.eye(K)
    
    # Innovation: y - H x
    # Use the mean of observations as the "true" observation vector
    y_mean = np.mean(observations, axis=0)
    
    # Innovation vector: (K,)
    innovation = y_mean - H @ mean
    
    # Innovation covariance: S = H C H^T + R
    S = H @ cov_loc @ H.T + R
    
    # Gain: K_gain = C H^T S^-1
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    gain = cov_loc @ H.T @ S_inv
    
    # Update: x_a = x_f + K (y - H x_f)
    mean_a = mean + gain @ innovation
    
    # For the ensemble, use perturbed observations
    innovations = observations - ensemble[:, observed_indices]  # (M, K)
    
    # Update each member
    result = ensemble + innovations @ gain.T
    
    # Adaptive Inflation:
    # Check if the ensemble spread is too small compared to the observation error
    ens_var_obs = np.var(ensemble[:, observed_indices], axis=0)  # (K,)
    obs_var = observation_variance
    
    # Compute a robust inflation factor
    spread_ratio = np.mean(ens_var_obs) / obs_var if obs_var > 0 else 1.0
    
    # Target spread ratio around 1.0-1.5
    target_ratio = 1.2
    if spread_ratio < target_ratio * 0.5:
        # Too under-dispersed, inflate more
        inflation = 1.0 + 0.15 * (target_ratio * 0.5 - spread_ratio) / target_ratio
    elif spread_ratio > target_ratio * 2.0:
        # Too over-dispersed, deflate slightly
        inflation = 1.0 - 0.05 * (spread_ratio - target_ratio * 2.0) / target_ratio
    else:
        # In the target range, mild inflation
        inflation = 1.0 + 0.02 * (target_ratio - spread_ratio) / target_ratio
    
    # Clamp inflation to reasonable bounds
    inflation = np.clip(inflation, 0.95, 1.3)
    
    # Apply inflation around the updated mean
    result = mean_a + inflation * (result - mean_a)
    
    # Ensure the result is finite and within bounds
    result = np.clip(result, -100, 100)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110:
        return result
    
    return result
