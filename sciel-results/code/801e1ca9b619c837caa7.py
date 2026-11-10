"""Ensemble Kalman Filter for Lorenz-96 assimilation."""
import numpy as np
import time

# Module-level state for adaptive inflation
_inflation_state = {'alpha': 1.0, 'target_infl': 0.0}

def _gaspari_cohn(d, c=3.0):
    """Gaspari-Cohn 5th-order correlation function for localization."""
    d = np.abs(d)
    result = np.zeros_like(d)
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    # d <= 0.5c
    if np.any(mask1):
        dm = d[mask1]
        cm = c
        result[mask1] = (-5.0/4.0)*(dm**5)/(cm**5) + 5.0*(dm**4)/(cm**4) - 10.0*(dm**3)/(cm**3) + 5.0*dm - 1.5
    # 0.5c < d <= 1.5c
    if np.any(mask2):
        dm = d[mask2]
        cm = c
        result[mask2] = (1.0/12.0)*(dm**5)/(cm**5) - (5.0/4.0)*(dm**4)/(cm**4) + (5.0/2.0)*(dm**3)/(cm**3) - 5.0*(dm**2)/(cm**2) + (25.0/12.0)*(dm/cm) + 1.0/2.0
    # 1.5c < d <= 2.5c
    if np.any(mask3):
        dm = d[mask3]
        cm = c
        result[mask3] = (-1.0/16.0)*(dm**5)/(cm**5) + (1.0/4.0)*(dm**4)/(cm**4) - (15.0/8.0)*(dm**3)/(cm**3) + (5.0/2.0)*(dm**2)/(cm**2) - (25.0/12.0)*(dm/cm) + 25.0/12.0
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis step for Lorenz-96.
    
    Args:
        ensemble: (16, 20) array of ensemble members
        observations: (10,) array of noisy observations of even-indexed variables
        observed_indices: indices of observed variables (even indices 0,2,4,...,18)
        observation_variance: scalar variance of observation noise
    
    Returns:
        updated ensemble with shape (16, 20)
    """
    deadline = time.time() + 110.0  # 110s safety margin
    
    N = ensemble.shape[0]  # 16
    dim = ensemble.shape[1]  # 20
    nobs = len(observations)  # 10
    
    # Work in a copy
    result = ensemble.copy()
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(result, axis=0)
    anomalies = result - ens_mean  # (16, 20)
    
    # Compute ensemble covariance in observation space
    # P_yy = (1/(N-1)) * Y_obs * Y_obs.T where Y_obs = anomalies[:, observed_indices]
    Y_obs = anomalies[:, observed_indices]  # (16, 10)
    
    # Observation error covariance (diagonal)
    R = observation_variance * np.eye(nobs)
    
    # Compute Kalman gain using the standard formula
    # P_yy = Y_obs @ Y_obs.T / (N-1)
    P_yy = (Y_obs @ Y_obs.T) / (N - 1)
    
    # Innovation covariance: S = P_yy + R
    S = P_yy + R
    
    # Innovation: y - H(x_mean)
    innovation = observations - ens_mean[observed_indices]
    
    # Compute gain: K = P_xy @ S^{-1}
    # P_xy = (1/(N-1)) * anomalies @ Y_obs.T  (20 x 10)
    P_xy = (anomalies @ Y_obs.T) / (N - 1)
    
    # Solve for gain: K = P_xy @ inv(S)
    # Use Cholesky for stability
    try:
        L = np.linalg.cholesky(S)
        # S^{-1} = L^{-T} @ L^{-1}
        # K = P_xy @ S^{-1}
        # Solve L @ X^T = P_xy.T for X^T, then K = X @ L^{-T}
        # Actually: K = P_xy @ inv(S)
        # inv(S) = inv(L^T) @ inv(L)
        # So K = P_xy @ inv(L^T) @ inv(L)
        # Let M = P_xy @ inv(L^T), then K = M @ inv(L)
        # Solve L^T @ M.T = P_xy.T  =>  M = solve(L^T, P_xy.T).T
        M = np.linalg.solve(L.T, P_xy.T).T  # (20, 10)
        K = np.linalg.solve(L, M)  # (20, 10)
    except np.linalg.LinAlgError:
        # Fallback to direct inverse
        try:
            S_inv = np.linalg.inv(S)
            K = P_xy @ S_inv
        except np.linalg.LinAlgError:
            # Last resort: simple nudging
            result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
            return np.clip(result, -100, 100)
    
    # Apply localization to the gain
    # Build localization matrix for full state space
    loc_matrix = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            # Cyclic distance on ring of 20
            d = abs(i - j)
            d = min(d, dim - d)
            loc_matrix[i, j] = _gaspari_cohn(d, c=3.0)
    
    # Localize the gain: K_loc = K * loc_matrix[observed_indices, :]
    # K is (20, 10), we need to multiply each row i of K by loc_matrix[i, observed_indices]
    K_loc = K * loc_matrix[observed_indices, :]  # (20, 10)
    
    # Update all ensemble members
    # x_new = x + K_loc @ (y - H(x_mean))
    update = K_loc @ innovation  # (20,)
    result = result + update[np.newaxis, :]
    
    # Adaptive inflation
    # Compute analysis innovation to detect under-dispersion
    ens_mean_analysis = np.mean(result, axis=0)
    innovation_analysis = observations - ens_mean_analysis[observed_indices]
    
    # Expected innovation variance
    P_yy_analysis = (anomalies[:, observed_indices] @ anomalies[:, observed_indices].T) / (N - 1)
    expected_innov_var = np.diag(P_yy_analysis) + observation_variance
    
    # Actual innovation variance
    actual_innov_var = np.mean(innovation_analysis ** 2)
    
    # Ratio to adjust inflation
    ratio = actual_innov_var / max(np.mean(expected_innov_var), 1e-10)
    
    # Smooth inflation adjustment
    target = 1.0 + 0.1 * (ratio - 1.0)
    target = max(1.0, min(target, 1.5))
    _inflation_state['alpha'] = 0.9 * _inflation_state['alpha'] + 0.1 * target
    
    # Apply inflation
    if _inflation_state['alpha'] > 1.0:
        new_mean = np.mean(result, axis=0)
        result = new_mean + _inflation_state['alpha'] * (result - new_mean)
    
    # Ensure finite and within bounds
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100, 100)
    
    return result
