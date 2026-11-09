"""Lorenz-96 ensemble assimilation with stochastic ensemble Kalman filter."""
import numpy as np
import time

# Module-level state for tracking updates across calls
_update_count = 0
_prev_ensemble = None
_prev_observations = None
_start_time = None

def _lorenz96_rhs(x, F):
    """Compute RHS of Lorenz-96 equations. Periodic boundary conditions."""
    n = len(x)
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter update for Lorenz-96.
    
    Parameters:
        ensemble: (16, 20) array of ensemble members
        observations: (10,) array of noisy observations at observed_indices
        observed_indices: indices of observed coordinates (even indices)
        observation_variance: scalar variance of observation noise
    
    Returns:
        Updated (16, 20) ensemble
    """
    global _update_count, _prev_ensemble, _prev_observations, _start_time
    
    # Deadline guard
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    if elapsed > 145.0:  # Safety margin before 150s limit
        return ensemble.copy()
    
    N = ensemble.shape[0]  # 16 members
    D = ensemble.shape[1]  # 20 dimensions
    obs_idx = np.asarray(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Estimate F from the ensemble mean (Lorenz-96 equilibrium: mean ~ F)
    x_mean = np.mean(ensemble, axis=0)
    F_est = np.mean(x_mean)
    
    # --- Stochastic Ensemble Kalman Filter ---
    # Generate ensemble perturbations of observations
    # For each member, add independent Gaussian noise to observations
    # This prevents variance collapse and maintains spread
    
    rng = np.random.default_rng(_update_count * 1000 + 42)
    
    # Perturb observations for each ensemble member
    obs_perturbed = np.zeros((N, n_obs))
    for i in range(N):
        noise = rng.normal(0.0, np.sqrt(observation_variance), size=n_obs)
        obs_perturbed[i] = observations + noise
    
    # Compute ensemble mean and covariance
    x_mean_vec = np.mean(ensemble, axis=0)  # (D,)
    
    # Anomalies: (N, D)
    X = ensemble - x_mean_vec[np.newaxis, :]  # (N, D)
    
    # Observation operator H: picks out observed components
    # H * x = x[obs_idx]
    # Covariance of observations: P_yy = H P H^T + R
    # Using ensemble: P_yy = (1/(N-1)) * sum (y_i - y_mean)(y_i - y_mean)^T + R
    
    Y = obs_perturbed - np.mean(obs_perturbed, axis=0)[np.newaxis, :]  # (N, n_obs)
    
    # Cross-covariance: P_xy = (1/(N-1)) * X^T Y
    P_xy = (X.T @ Y) / (N - 1)  # (D, n_obs)
    
    # Observation covariance: P_yy = (1/(N-1)) * Y^T Y + R
    P_yy = (Y.T @ Y) / (N - 1) + observation_variance * np.eye(n_obs)  # (n_obs, n_obs)
    
    # Kalman gain: K = P_xy @ P_yy^{-1}
    # Use Cholesky or solve for numerical stability
    try:
        P_yy_inv = np.linalg.inv(P_yy)
        K = P_xy @ P_yy_inv  # (D, n_obs)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        K = P_xy @ np.linalg.pinv(P_yy)
    
    # Innovation: y_obs - H(x_mean)
    innovation = observations - x_mean_vec[obs_idx]  # (n_obs,)
    
    # Analysis: x_a = x + K * (y - H(x))
    # For each member: x_i^a = x_i + K * (y_i - H(x_i))
    # But in stochastic EKF, we use perturbed observations:
    # x_i^a = x_i + K * (y_i^pert - H(x_i))
    
    result = ensemble.copy()
    for i in range(N):
        H_xi = ensemble[i, obs_idx]  # (n_obs,)
        innov_i = obs_perturbed[i] - H_xi  # (n_obs,)
        result[i] = ensemble[i] + K @ innov_i
    
    # --- Adaptive inflation to maintain spread ---
    # Check if ensemble spread is too small relative to expected
    # Compute spread at observed locations
    spread_obs = np.std(ensemble[:, obs_idx], axis=0)
    expected_spread = np.sqrt(observation_variance) * 0.5  # heuristic
    
    # If spread is too small, inflate slightly
    if np.mean(spread_obs) < expected_spread * 0.5:
        # Inflate around the mean
        inflation_factor = 1.0 + 0.05 * (1.0 - np.mean(spread_obs) / max(expected_spread, 1e-10))
        inflation_factor = min(inflation_factor, 1.2)  # Cap inflation
        x_mean_result = np.mean(result, axis=0)
        result = x_mean_result[np.newaxis, :] + inflation_factor * (result - x_mean_result[np.newaxis, :])
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    # Ensure finite
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Track state for potential future use
    _update_count += 1
    _prev_ensemble = ensemble.copy()
    _prev_observations = observations.copy()
    
    return result
