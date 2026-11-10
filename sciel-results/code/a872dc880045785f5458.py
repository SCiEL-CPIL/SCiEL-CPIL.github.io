"""Lorenz-96 ensemble assimilation: localized ensemble Kalman filter with adaptive inflation."""
import numpy as np
import time

# Pre-compute the Gaspari-Cohn correlation function for localization
def _gaspari_cohn(distance, L=6.0):
    """Gaspari-Cohn 5th order polynomial correlation function."""
    d = np.abs(distance)
    L = float(L)
    a = 5.0 / L
    b = 1.0 / L
    result = np.zeros_like(d, dtype=float)
    mask1 = d <= L / 2.0
    mask2 = (d > L / 2.0) & (d <= L)
    # For d <= L/2
    x1 = d[mask1]
    result[mask1] = (-1/4)*(a*x1)**5 + (5/2)*(a*x1)**4 - (5/2)*(a*x1)**3 + (5/12)
    # For L/2 < d <= L
    x2 = d[mask2]
    result[mask2] = (1/2)*(a*x2 - 2)**5 - (5/2)*(a*x2 - 2)**4 + (5/2)*(a*x2 - 2)**3 + (5/3)*(a*x2 - 2) + 4/3
    # For d > L, result is 0 (already initialized)
    return result

# Pre-compute localization matrix for 20 variables with periodic boundary
_N = 20
_distances = np.array([min(abs(i - j), _N - abs(i - j)) for i in range(_N) for j in range(_N)]).reshape(_N, _N)
_LOCALIZATION = _gaspari_cohn(_distances, L=5.0)
_LOCALIZATION = 0.5 * (_LOCALIZATION + _LOCALIZATION.T)  # Symmetrize
np.fill_diagonal(_LOCALIZATION, 1.0)


def _lorenz96(x, F=8.0, dt=0.01):
    """Propagate Lorenz-96 state by one RK4 step."""
    n = len(x)
    def f(state):
        return (state[np.arange(1, n)] - state[(np.arange(n) - 2) % n]) * state[(np.arange(n) - 1) % n] - state + F
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized Ensemble Kalman Filter update with adaptive inflation.
    
    Parameters:
        ensemble: (16, 20) array of ensemble members
        observations: (10,) array of noisy observations of even-indexed variables
        observed_indices: array of observed variable indices
        observation_variance: scalar observation variance
    
    Returns:
        updated ensemble with same (16, 20) shape
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n_members, n_vars = ensemble.shape
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    
    # Ensure observations is 1D
    obs = np.asarray(observations, dtype=float).ravel()
    
    # Compute ensemble mean and anomalies
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean[None, :]  # (n_members, n_vars)
    
    # Compute ensemble covariance at observed locations
    # P_oo = (1/(N-1)) * Y_oo @ Y_oo.T where Y_oo is anomalies at obs locations
    Y_oo = anomalies[:, obs_idx]  # (n_members, n_obs)
    
    # Observation covariance
    R = observation_variance * np.eye(n_obs)
    
    # Compute Kalman gain using ensemble formulation
    # S = Y_oo @ Y_oo.T / (N-1) + R
    S = Y_oo @ Y_oo.T / (n_members - 1) + R
    
    # K = Y @ S^{-1} where Y is full anomaly matrix
    # Use Cholesky or direct solve
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain: (n_vars, n_obs)
    K = anomalies @ S_inv  # (n_members, n_obs) -> we need (n_vars, n_obs)
    # Actually: K = P_xo * S^{-1}, where P_xo = Y @ Y_oo.T / (N-1)
    P_xo = anomalies @ Y_oo.T / (n_members - 1)  # (n_vars, n_obs)
    K = P_xo @ S_inv  # (n_vars, n_obs)
    
    # Innovation
    innovation = obs - mean[obs_idx]  # (n_obs,)
    
    # Apply localization to the gain
    # Localize by scaling K rows with the localization function
    K_localized = K * _LOCALIZATION  # Element-wise scaling
    
    # Update ensemble mean
    new_mean = mean + K_localized @ innovation  # (n_vars,)
    
    # Update anomalies using the square root filter approach
    # For each member: a_i' = a_i - K_localized @ (Y_oo_i - obs_noise_correction)
    # Actually use the standard ensemble update:
    # a_i' = a_i - K @ (y_i - y_obs_mean_adjusted)
    # where y_i = Y_oo[i] and we use the ensemble-based formula
    
    # Standard ensemble Kalman update:
    # a_i' = a_i - K @ (Y_oo_i - obs + obs - mean_obs)
    # = a_i - K @ Y_oo_i + K @ (obs - mean_obs)
    # But with localization, we use a different approach
    
    # Let's use the direct ensemble update with localized gain:
    # a_i' = a_i - K_loc @ (Y_oo_i - (obs - mean_obs))
    # This is equivalent to: a_i' = a_i - K_loc @ Y_oo_i + K_loc @ innovation
    
    Y_oo_i = Y_oo  # (n_members, n_obs)
    correction = K_localized.T @ Y_oo_i  # (n_obs, n_members) -> transpose
    # Actually: correction_i = K_loc @ Y_oo_i for each i
    # K_loc is (n_vars, n_obs), Y_oo_i is (n_obs,) for member i
    # So correction matrix = Y_oo.T @ K_loc.T = (n_members, n_obs) @ (n_obs, n_vars)
    corrections = Y_oo @ K_localized.T  # (n_members, n_vars)
    
    new_anomalies = anomalies - corrections + (K_localized @ innovation)[None, :]  # (n_members, n_vars)
    
    # Adaptive inflation: check if ensemble spread is too small
    # Compute spread at observed locations
    obs_spread = np.sqrt(np.sum(new_anomalies[:, obs_idx]**2, axis=0) / (n_members - 1))
    obs_var = observation_variance
    # If spread is too small relative to observation uncertainty, inflate
    spread_ratio = np.mean(obs_spread) / np.sqrt(obs_var)
    if spread_ratio < 0.7:
        inflation_factor = 1.0 + 0.1 * (0.7 - spread_ratio)
        new_anomalies = new_anomalies * inflation_factor
    
    # Reconstruct ensemble
    result = new_mean[None, :] + new_anomalies
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    # Ensure finite
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline check
    if time.time() > deadline:
        return result
    
    return result
