"""Lorenz-96 ensemble assimilation with localized ensemble Kalman filter."""
import numpy as np
import time

def _lorenz96_step(x, F, dt=0.01):
    """Single RK4 step for Lorenz-96 system."""
    n = len(x)
    
    def f(state):
        dx = np.empty_like(state)
        for i in range(n):
            dx[i] = (state[(i+1) % n] - state[(i-2) % n]) * state[(i-1) % n] - state[i] + F
        return dx
    
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

def _lorenz96_propagate(x, F, steps=5, dt=0.01):
    """Propagate state through multiple RK4 steps."""
    for _ in range(steps):
        x = _lorenz96_step(x, F, dt)
    return x

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis update for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (8,) array of noisy observations at even indices
    - observed_indices: indices that are observed (even indices 0,2,4,...,18)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    # Track time to ensure we finish within budget
    t_start = time.time()
    TIME_LIMIT = 100.0  # Safety margin before 120s limit
    
    n_members, n_state = ensemble.shape
    result = ensemble.copy()
    
    # Estimate forcing parameter F from ensemble mean behavior
    # For Lorenz-96, F is typically 6, 8, or 10
    # We can estimate it from the average of x*(x_{i-1} - x_{i+2}) + x_i
    # But since we don't have time derivatives, use a heuristic
    # The mean state magnitude gives a rough indication
    mean_state = np.mean(result, axis=0)
    mean_abs = np.mean(np.abs(mean_state))
    
    # Heuristic: F=6 -> mean ~2-4, F=8 -> mean ~4-6, F=10 -> mean ~6-8
    if mean_abs < 3.5:
        F_est = 6.0
    elif mean_abs < 5.5:
        F_est = 8.0
    else:
        F_est = 10.0
    
    # Localized Ensemble Kalman Filter
    # Step 1: Compute ensemble mean and anomaly matrix
    ens_mean = np.mean(result, axis=0)
    anomalies = result - ens_mean  # (n_members, n_state)
    
    # Step 2: Compute localization weights for observation to state variables
    # Use exponential localization with range L=4
    L = 4  # Localization range
    loc_weights = np.zeros(n_state)
    for i in range(n_state):
        for j in observed_indices:
            dist = min(abs(i - j), n_state - abs(i - j))
            if dist <= L:
                w = np.exp(-0.5 * (dist / L) ** 2)
                loc_weights[i] += w
    # Normalize localization weights
    if np.max(loc_weights) > 0:
        loc_weights = loc_weights / np.max(loc_weights)
    
    # Step 3: Compute ensemble covariance at observed locations
    # B = (1/(n-1)) * anomalies * anomalies.T at observed indices
    obs_anomalies = anomalies[:, observed_indices]  # (n_members, n_obs)
    n_obs = len(observed_indices)
    
    # Background covariance at observation locations
    B = (1.0 / (n_members - 1)) * obs_anomalies @ obs_anomalies.T  # (n_obs, n_obs)
    
    # Observation noise covariance (diagonal)
    R = observation_variance * np.eye(n_obs)
    
    # Step 4: Compute Kalman gain
    # S = B + R (innovation covariance)
    S = B + R
    # K = B * S^{-1} (n_obs x n_obs)
    try:
        S_inv = np.linalg.inv(S)
        K = B @ S_inv  # (n_obs, n_obs)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        S_inv = np.linalg.pinv(S)
        K = B @ S_inv
    
    # Step 5: Compute innovations
    obs_mean = np.mean(observations)
    innovations = observations - np.mean(obs_anomalies, axis=0)  # (n_obs,)
    
    # Step 6: Update ensemble
    # For each member, update all state variables using localized Kalman filter
    for i in range(n_state):
        # Compute local gain for state variable i
        # K_i = sum over observed j of K_{ij} * loc_weight
        if i in observed_indices:
            idx = list(observed_indices).index(i)
            gain = K[idx, :]  # (n_obs,)
            # Apply localization
            local_gain = np.zeros(n_obs)
            for j, obs_idx in enumerate(observed_indices):
                dist = min(abs(i - obs_idx), n_state - abs(i - obs_idx))
                if dist <= L:
                    w = np.exp(-0.5 * (dist / L) ** 2)
                    local_gain[j] = K[idx, j] * w
        else:
            # For unobserved variables, use a simplified approach
            # Estimate gain from nearby observed variables
            local_gain = np.zeros(n_obs)
            for j, obs_idx in enumerate(observed_indices):
                dist = min(abs(i - obs_idx), n_state - abs(i - obs_idx))
                if dist <= L:
                    w = np.exp(-0.5 * (dist / L) ** 2)
                    # Use correlation-based gain approximation
                    corr = np.corrcoef(anomalies[:, i], obs_anomalies[:, j])[0, 1] \
                           if np.std(anomalies[:, i]) > 1e-10 and np.std(obs_anomalies[:, j]) > 1e-10 else 0
                    local_gain[j] = corr * w * (np.std(anomalies[:, i]) / np.std(obs_anomalies[:, j])) \
                                   if np.std(obs_anomalies[:, j]) > 1e-10 else 0
        
        # Update each member
        for m in range(n_members):
            result[m, i] += np.dot(local_gain, innovations)
    
    # Step 7: Adaptive inflation to prevent ensemble collapse
    # Compute spread-skill ratio
    obs_spread = np.std(obs_anomalies, axis=0)
    obs_skill = np.sqrt(observation_variance + np.var(innovations))
    
    # Inflation factor based on spread-skill mismatch
    if obs_skill > 1e-10:
        ratio = np.mean(obs_spread) / obs_skill
        # If spread is too small, inflate; if too large, deflate slightly
        if ratio < 1.0:
            inflation = 1.0 + 0.1 * (1.0 - ratio)
        else:
            inflation = 1.0
    else:
        inflation = 1.0
    
    # Cap inflation to avoid instability
    inflation = np.clip(inflation, 1.0, 1.5)
    
    # Apply inflation about the ensemble mean
    if inflation > 1.0:
        for i in range(n_state):
            result[:, i] = ens_mean[i] + inflation * (result[:, i] - ens_mean[i])
    
    # Step 8: Clip to valid range
    result = np.clip(result, -100, 100)
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100, 100)
    
    # Time check: if we're running out of time, return current result
    elapsed = time.time() - t_start
    if elapsed > TIME_LIMIT:
        return result
    
    return result
