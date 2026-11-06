"""Ensemble analysis update for Lorenz-96 with localized EnKF and inflation."""
import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an ensemble analysis update using a localized Ensemble Kalman Filter
    with multiplicative inflation and rank-deficiency handling.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array_like
        Indices of observed variables.
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    ndarray, shape (16, 20)
        Updated ensemble.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_members, n_vars = ensemble.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # Compute ensemble mean and anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean  # shape (n_members, n_vars)
    
    # Compute ensemble covariance (with bias correction)
    # C = anomalies.T @ anomalies / (n_members - 1)
    # But we'll work with the anomaly matrix directly for efficiency
    
    # Localized covariance for observed variables
    # Use a simple exponential localization kernel
    # For Lorenz-96, a localization radius of about 3-5 grid points works well
    
    def localization_weight(i, j, radius=4.0):
        """Exponential localization weight."""
        d = abs(i - j)
        # Periodic distance for cyclic Lorenz-96
        d = min(d, n_vars - d)
        return np.exp(-d / radius)
    
    # Build localized covariance for observations
    # We need C_yy = H P H^T where H is the observation operator
    # P = anomalies.T @ anomalies / (n_members - 1)
    
    # For efficiency with small ensemble, use the ensemble-based approach
    # The Kalman gain can be computed in the ensemble anomaly space
    
    # Observation operator: extracts observed indices
    H = np.zeros((n_obs, n_vars))
    for k, idx in enumerate(obs_idx):
        H[k, idx] = 1.0
    
    # Localize the observation-space covariance
    # C_yy_localized = H @ P_localized @ H^T
    
    # Compute P_localized by applying localization to the full covariance
    # For a 20-variable system, we can do this directly
    P = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Apply localization to P
    P_loc = np.zeros_like(P)
    for i in range(n_vars):
        for j in range(n_vars):
            P_loc[i, j] = P[i, j] * localization_weight(i, j, radius=3.0)
    
    # Observation-space covariance
    C_yy = H @ P_loc @ H.T
    
    # Add observation noise
    R = observation_variance * np.eye(n_obs)
    S = C_yy + R  # Innovation covariance
    
    # Innovation
    y_obs = observations - H @ mean  # shape (n_obs,)
    
    # Kalman gain
    # K = P_loc @ H^T @ S^(-1)
    # For numerical stability, use Cholesky or pseudo-inverse
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = P_loc @ H.T @ S_inv  # shape (n_vars, n_obs)
    
    # Analysis update
    # mean_a = mean + K @ (y_obs)
    # anomalies_a = (I - K @ H) @ anomalies
    
    mean_a = mean + K @ y_obs
    KH = K @ H  # shape (n_vars, n_vars)
    anomalies_a = (np.eye(n_vars) - KH) @ anomalies.T  # shape (n_vars, n_members)
    anomalies_a = anomalies_a.T  # back to (n_members, n_vars)
    
    ensemble_a = mean_a + anomalies_a
    
    # Multiplicative inflation to counteract ensemble collapse
    # Compute the spread of the analysis ensemble
    spread_a = np.std(ensemble_a, axis=0)
    spread_f = np.std(ensemble, axis=0)
    
    # Target spread: slightly larger than forecast spread
    # Use a simple adaptive inflation
    alpha = 1.0
    for i in range(n_vars):
        if spread_f[i] > 1e-10:
            ratio = spread_a[i] / spread_f[i]
            if ratio < 0.8:
                alpha = max(alpha, 1.0 / np.sqrt(ratio))
    
    # Cap inflation to avoid divergence
    alpha = min(alpha, 1.5)
    
    # Apply inflation
    mean_final = np.mean(ensemble_a, axis=0)
    anomalies_final = ensemble_a - mean_final
    ensemble_final = mean_final + alpha * anomalies_final
    
    # Ensure finite values and within bounds
    ensemble_final = np.nan_to_num(ensemble_final, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble_final = np.clip(ensemble_final, -100.0, 100.0)
    
    return ensemble_final
