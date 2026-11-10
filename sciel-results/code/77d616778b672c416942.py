"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gaspari_cohn(distance, L):
    """Gaspari-Cohn correlation function."""
    distance = np.abs(distance)
    L = max(L, 1e-10)
    # Normalized distance
    d = distance / L
    result = np.zeros_like(d)
    mask1 = d <= 1.0
    mask2 = (d > 1.0) & (d <= 2.0)
    
    # For d <= 1
    if np.any(mask1):
        dm = d[mask1]
        result[mask1] = (-0.25 * dm**5 + 0.5 * dm**4 + 0.625 * dm**3 - 0.5 * dm**2 + 1.0)
    
    # For 1 < d <= 2
    if np.any(mask2):
        dm = d[mask2]
        result[mask2] = (0.0625 * dm**5 - 0.125 * dm**4 + 0.625 * dm**3 - 0.5 * dm**2 + 0.125 * dm - 0.0625)
    
    # For d > 2, result stays 0
    return result


def _local_esrf_update(ensemble, observations, observed_indices, observation_variance, 
                        localization_length=4.0, inflation_factor=1.01):
    """
    Localized Ensemble Square Root Filter (ESRF) update.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (N, D)
        Ensemble of N members, each of dimension D.
    observations : ndarray, shape (N_obs,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed state variables.
    observation_variance : float
        Variance of observation noise.
    localization_length : float
        Length scale for Gaspari-Cohn localization.
    inflation_factor : float
        Multiplicative inflation factor to prevent collapse.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (N, D)
    """
    N, D = ensemble.shape
    result = ensemble.copy()
    
    # Compute ensemble mean
    ens_mean = np.mean(ensemble, axis=0)
    
    # Compute ensemble covariance (anomaly matrix)
    anomalies = ensemble - ens_mean  # shape (N, D)
    
    # Observation indices
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # Observation operator: extract observed components
    H = np.zeros((n_obs, D))
    for i, idx in enumerate(obs_idx):
        H[i, idx] = 1.0
    
    # Observation vector (same for all members)
    y = np.array(observations, dtype=float)
    
    # Compute forecast observation anomaly
    H_anomalies = anomalies[:, obs_idx]  # shape (N, n_obs)
    
    # Innovation
    innovation = y - H_anomalies  # shape (N, n_obs) -- actually we want (n_obs,) mean innovation
    
    # For ESRF, we work with the anomaly form
    # Observation error covariance R = observation_variance * I
    R = observation_variance * np.eye(n_obs)
    
    # Compute the gain using the ensemble-based approach
    # We'll use the standard EnKF/ESRF formulation with localization
    
    # Localized update: for each state variable, compute the localized gain
    # and apply the update
    
    # Precompute the inverse of (H S H^T + R) where S is the ensemble covariance
    # S = (1/(N-1)) * anomalies^T * anomalies
    # H S H^T = (1/(N-1)) * H_anomalies^T * H_anomalies
    
    # To avoid numerical issues, we use the innovation-based formulation
    # y_anom = y - H * ens_mean
    y_anom = y - H.dot(ens_mean)  # shape (n_obs,)
    
    # Covariance of forecast observations
    # S_yy = (1/(N-1)) * H_anomalies^T @ H_anomalies
    S_yy = (H_anomalies.T @ H_anomalies) / (N - 1)  # shape (n_obs, n_obs)
    
    # Add observation noise
    S_yy_R = S_yy + R  # shape (n_obs, n_obs)
    
    # Compute the gain: K = S_xy * inv(S_yy_R)
    # where S_xy = (1/(N-1)) * anomalies^T @ H_anomalies
    # K has shape (D, n_obs)
    
    # For localization, we apply the Gaspari-Cohn function to the cross-covariance
    # S_xy[i, j] corresponds to state i and observation j
    # We localize in the state dimension
    
    S_xy = (anomalies.T @ H_anomalies) / (N - 1)  # shape (D, n_obs)
    
    # Apply localization to S_xy
    # For each state i and observation j, the distance is |i - obs_idx[j]|
    localization_matrix = np.zeros((D, n_obs))
    for j, obs_j in enumerate(obs_idx):
        distances = np.abs(np.arange(D) - obs_j)
        localization_matrix[:, j] = _gaspari_cohn(distances, localization_length)
    
    S_xy_local = S_xy * localization_matrix  # localized cross-covariance
    
    # Compute gain
    # K = S_xy_local @ inv(S_yy_R)
    try:
        S_yy_R_inv = np.linalg.inv(S_yy_R)
    except np.linalg.LinAlgError:
        S_yy_R_inv = np.linalg.pinv(S_yy_R)
    
    K = S_xy_local @ S_yy_R_inv  # shape (D, n_obs)
    
    # Update each ensemble member
    # For ESRF, we use the square-root formulation to preserve spread
    # x_a = x_f + K * (y - H x_f)
    # But to maintain the ensemble spread properly, we use the following:
    
    # First, compute the analysis mean
    ens_mean_analysis = ens_mean + K.dot(y_anom)  # shape (D,)
    
    # For the ESRF update of individual members, we use the following approach:
    # We want to update the anomalies such that the new covariance is consistent
    # with the analysis covariance.
    
    # The analysis covariance is:
    # S_a = S_f - K * H * S_f
    # In anomaly form: A_a = A_f * (I - K * H * C_f)^{-1/2} ... this gets complex.
    
    # A simpler approach: use the standard EnKF update with localization,
    # then apply inflation to prevent collapse.
    
    # Standard EnKF update:
    # x_a_i = x_f_i + K * (y - H x_f_i)
    
    updated_ensemble = np.zeros_like(ensemble)
    for i in range(N):
        H_xf_i = H.dot(ensemble[i])  # forecast observation for member i
        innovation_i = y - H_xf_i  # shape (n_obs,)
        updated_ensemble[i] = ensemble[i] + K.dot(innovation_i)
    
    # Apply multiplicative inflation to prevent ensemble collapse
    # Inflate the anomalies around the analysis mean
    anomalies_analysis = updated_ensemble - ens_mean_analysis
    updated_ensemble = ens_mean_analysis + inflation_factor * anomalies_analysis
    
    return updated_ensemble


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an ensemble analysis update using a localized ESRF.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of 16 members, each a 20-dimensional state.
    observations : ndarray, shape (10,)
        Noisy observations of the 10 even-indexed state variables.
    observed_indices : array-like
        Indices of observed state variables (should be [0, 2, 4, ..., 18]).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    try:
        updated = _local_esrf_update(
            ensemble, 
            observations, 
            observed_indices, 
            observation_variance,
            localization_length=4.0,
            inflation_factor=1.01
        )
    except Exception:
        # Fallback to simple nudging if something goes wrong
        updated = ensemble.copy()
        obs_idx = np.array(observed_indices)
        updated[:, obs_idx] += 0.5 * (observations - updated[:, obs_idx])
    
    # Ensure output is finite and within bounds
    updated = np.clip(updated, -100.0, 100.0)
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Check deadline
    if time.time() > deadline:
        # Return the best result we have
        pass
    
    return updated

# EVOLVE-BLOCK-END
