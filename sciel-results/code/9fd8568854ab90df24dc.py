"""Rewritten program for Lorenz-96 assimilation with ensemble Kalman filter."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x has shape (N, 20) or (20,)."""
    if x.ndim == 1:
        x_1 = np.roll(x, 1)
        x_m1 = np.roll(x, -1)
        x_m2 = np.roll(x, 2)
        return (x_1 - x_m2) * x_m1 - x + F
    else:
        x_1 = np.roll(x, 1, axis=1)
        x_m1 = np.roll(x, -1, axis=1)
        x_m2 = np.roll(x, 2, axis=1)
        return (x_1 - x_m2) * x_m1 - x + F


def _rk4_step(x, F, dt, n_steps=1):
    """Propagate ensemble using RK4 for n_steps."""
    x = x.copy()
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def _localization_weight(dist, decay_scale):
    """Exponential localization weight."""
    return np.exp(-dist / decay_scale)


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis update for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (16, 10) array of noisy observations of even-indexed coordinates
    - observed_indices: array of observed coordinate indices (even indices: 0,2,...,18)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Ensemble mean and covariance
    x_mean = np.mean(ensemble, axis=0)  # (20,)
    
    # Centered anomalies
    X = ensemble - x_mean  # (16, 20)
    
    # Sample covariance
    S = X.T @ X / (N - 1)  # (20, 20)
    
    # Observation operator: H selects observed indices
    # H is (n_obs, 20)
    H = np.zeros((n_obs, dim))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Innovation: y - H*x_mean
    y = observations.mean(axis=0)  # (10,) - mean of observations across ensemble members
    innovation = y - H @ x_mean  # (10,)
    
    # Observation covariance
    R = np.eye(n_obs) * observation_variance
    
    # Forecast observation covariance: H*S*H.T + R
    HSH = H @ S @ H.T  # (10, 10)
    P_y = HSH + R  # (10, 10)
    
    # Compute Kalman gain: K = S*H.T*P_y^{-1}
    try:
        P_y_inv = np.linalg.inv(P_y)
        K = S @ H.T @ P_y_inv  # (20, 10)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        P_y_inv = np.linalg.pinv(P_y)
        K = S @ H.T @ P_y_inv
    
    # Apply localization to the Kalman gain
    # Localize based on distance in the cyclic Lorenz-96 space
    decay_scale = 3.0  # Localization scale
    for i in range(dim):
        for j in range(n_obs):
            obs_idx = observed_indices[j]
            # Cyclic distance
            dist = min(abs(i - obs_idx), dim - abs(i - obs_idx))
            K[i, j] *= _localization_weight(dist, decay_scale)
    
    # Update ensemble
    result = ensemble.copy()
    for n in range(N):
        result[n] += K @ innovation
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    # Optional: add a small inflation to prevent ensemble collapse
    # This helps maintain spread in chaotic systems
    result_mean = np.mean(result, axis=0)
    inflation_factor = 1.001  # Small multiplicative inflation
    result = result_mean + inflation_factor * (result - result_mean)
    
    # Ensure finiteness and bounds
    result = np.where(np.isfinite(result), result, 0.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
