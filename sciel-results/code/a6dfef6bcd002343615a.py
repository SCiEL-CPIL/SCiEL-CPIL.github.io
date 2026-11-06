"""Ensemble Kalman Filter with adaptive inflation and localization for Lorenz-96 assimilation."""
import numpy as np
import time

def _lorenz96_rhs(x, F):
    """Compute RHS of Lorenz-96 system."""
    n = len(x)
    dx = np.zeros(n)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using an Ensemble Kalman Filter with:
    - Adaptive multiplicative inflation
    - Cosine-based spatial localization
    - Rank-deficiency handling via covariance regularization
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array_like
        Indices of observed variables.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after analysis step.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    n_members, n_vars = ensemble.shape
    result = ensemble.copy()
    
    # Convert to arrays
    obs_idx = np.asarray(observed_indices)
    obs = np.asarray(observations, dtype=float)
    R = float(observation_variance)
    
    # Compute ensemble mean and anomalies
    mean = np.mean(result, axis=0)
    anomalies = result - mean  # shape (16, 20)
    
    # Estimate background covariance (with regularization)
    # Use ensemble covariance: C = A^T A / (n-1)
    # Regularize to handle rank deficiency
    C = (anomalies.T @ anomalies) / (n_members - 1)
    # Add small regularization
    C += 1e-6 * np.eye(n_vars)
    
    # Apply spatial localization to covariance
    # Cosine localization: L(i,j) = cos(pi/2 * |i-j|/r) if |i-j| < r, else 0
    # Use a localization radius of about half the system size
    loc_radius = 8  # in indices
    loc_matrix = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            dist = abs(i - j)
            # Account for periodicity
            dist = min(dist, n_vars - dist)
            if dist < loc_radius:
                loc_matrix[i, j] = np.cos(0.5 * np.pi * dist / loc_radius)
    
    # Localized covariance
    C_loc = C * loc_matrix
    
    # Compute observation operator H (selects observed indices)
    H = np.zeros((len(obs_idx), n_vars))
    for k, idx in enumerate(obs_idx):
        H[k, int(idx)] = 1.0
    
    # Observation covariance
    R_mat = R * np.eye(len(obs_idx))
    
    # Innovation covariance: S = H C_loc H^T + R
    HC = H @ C_loc
    S = HC @ H.T + R_mat
    
    # Compute gain: K = C_loc H^T S^{-1}
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = C_loc @ H.T @ S_inv  # shape (n_vars, len(obs_idx))
    
    # Compute innovation: y - H x_bar
    innovation = obs - H @ mean  # shape (len(obs_idx),)
    
    # Analysis mean update
    mean_a = mean + K @ innovation
    
    # Analysis anomalies update (perturbation form)
    # For each member: x_a = x + K (y - H x)
    # In perturbation form: x_a - mean_a = (I - K H) (x - mean)
    I_KH = np.eye(n_vars) - K @ H
    anomalies_a = I_KH @ anomalies.T  # shape (n_vars, n_members)
    
    # Adaptive inflation
    # Compute analysis ensemble spread at observed locations
    obs_anomalies_a = anomalies_a[:, observed_indices] if hasattr(observed_indices, '__len__') else anomalies_a[:, obs_idx]
    analysis_spread = np.var(obs_anomalies_a, axis=0)  # variance at observed points
    
    # Compare to observation error
    # Inflate if analysis spread is too small relative to observation error
    target_spread = R  # we want analysis spread ~ R at observed points
    inflation_factor = 1.0
    
    # Simple adaptive inflation
    if np.mean(analysis_spread) < 0.5 * R:
        # Under-dispersed: inflate
        inflation_factor = 1.0 + 0.05 * (1.0 - np.mean(analysis_spread) / max(R, 1e-10))
        inflation_factor = min(inflation_factor, 1.15)
    elif np.mean(analysis_spread) > 2.0 * R:
        # Over-dispersed: slightly deflate
        inflation_factor = 1.0 - 0.02 * (np.mean(analysis_spread) / max(R, 1e-10) - 1.0)
        inflation_factor = max(inflation_factor, 0.95)
    
    # Apply inflation to anomalies
    anomalies_a *= inflation_factor
    
    # Reconstruct ensemble
    result = mean_a[None, :] + anomalies_a.T
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        # Fallback: simple nudging
        result = ensemble.copy()
        result[:, obs_idx] += 0.5 * (obs - result[:, obs_idx])
        result = np.clip(result, -100.0, 100.0)
    
    return result
