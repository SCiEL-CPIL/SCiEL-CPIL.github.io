"""Lorenz-96 ensemble assimilation combining EKF forecast with LESRF refinement."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 dynamics: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F"""
    n = len(x)
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _rk4_propagate_ensemble(ens, F, dt, steps):
    """Propagate an ensemble (n_members, n_vars) through multiple RK4 steps."""
    n_members = ens.shape[0]
    for _ in range(steps):
        for m in range(n_members):
            ens[m] = _rk4_step(ens[m], F, dt)
    return ens

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : (16, 20) array, current ensemble members
    observations : (10,) array, noisy observations of observed_indices
    observed_indices : indices that are observed (even indices: 0,2,4,...,18)
    observation_variance : scalar, variance of observation noise
    
    Returns:
    --------
    Updated ensemble with same (16, 20) shape.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_members, n_vars = ensemble.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # Convert to float
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    R_val = float(observation_variance)
    
    # Infer F from ensemble statistics
    ens_mean = np.mean(ens, axis=0)
    F_est = np.mean(ens_mean - (np.roll(ens_mean, -1) - np.roll(ens_mean, 2)) * np.roll(ens_mean, 1))
    F = np.clip(F_est, 4.0, 12.0)
    
    # Step 1: Forecast the ensemble forward 5 steps (dt=0.01 each)
    dt = 0.01
    n_forecast = 5
    ens = _rk4_propagate_ensemble(ens, F, dt, n_forecast)
    
    # Step 2: EKF bulk update
    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean  # (n_members, n_vars)
    
    # Observed components of anomalies
    H_anom = anomalies[:, obs_idx]  # (n_members, n_obs)
    
    # Innovation
    y_forecast = ens_mean[obs_idx]
    innovation = obs - y_forecast  # (n_obs,)
    
    # S = H P H^T + R
    alpha = n_members / (n_members - 1.0)
    HP = alpha * (H_anom @ H_anom.T) / (n_members - 1.0)  # (n_obs, n_obs)
    S = HP + R_val * np.eye(n_obs)
    
    # Compute inverse of S (10x10, cheap)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain K = P H^T S^-1
    PHt = alpha * (anomalies @ H_anom.T) / (n_members - 1.0)  # (n_vars, n_obs)
    K = PHt @ S_inv  # (n_vars, n_obs)
    
    # Analysis mean
    mean_analysis = ens_mean + K @ innovation
    
    # Perturbation update: x_a = mean + (I - KH) * anomaly
    # KH is (n_vars, n_vars) but sparse (only obs_idx columns non-zero)
    KH = np.zeros((n_vars, n_vars))
    KH[:, obs_idx] = K
    
    ens_analysis = np.empty_like(ens)
    for m in range(n_members):
        ens_analysis[m] = mean_analysis + (anomalies[m] - KH @ anomalies[m])
    
    # Step 3: LESRF refinement pass
    # Apply a sequential local ensemble square-root filter for additional accuracy
    ens_mean2 = np.mean(ens_analysis, axis=0)
    anomalies2 = ens_analysis - ens_mean2
    
    # Localization weights: Gaussian correlation with length ~5
    corr_length = 5.0
    loc_weights = np.zeros((n_obs, n_vars))
    for j, obs_i in enumerate(obs_idx):
        for i in range(n_vars):
            dist = abs(i - obs_i)
            dist = min(dist, n_vars - dist)
            loc_weights[j, i] = np.exp(-0.5 * (dist / corr_length) ** 2)
    
    # Sequential update for each observation
    for j in range(n_obs):
        obs_j = obs[j]
        obs_idx_j = obs_idx[j]
        w = loc_weights[j, :]
        
        if w[obs_idx_j] < 0.01:
            continue
        
        H_ens = ens_analysis[:, obs_idx_j]
        H_mean = np.mean(H_ens)
        H_anom_j = H_ens - H_mean
        
        # S for this local observation
        S_j = alpha * np.dot(H_anom_j, H_anom_j) / (n_members - 1.0) + R_val
        
        # Cross-covariance with localization
        for i in range(n_vars):
            if w[i] < 0.01:
                continue
            x_anom_i = anomalies2[:, i]
            cross_cov = alpha * np.dot(x_anom_i, H_anom_j) / (n_members - 1.0)
            cross_cov_loc = cross_cov * w[i]
            ens_analysis[:, i] += (cross_cov_loc / S_j) * (obs_j - H_ens)
        
        # Recompute anomalies
        ens_mean2 = np.mean(ens_analysis, axis=0)
        anomalies2 = ens_analysis - ens_mean2
    
    # Step 4: Inflation to prevent filter degeneration
    ens_mean_final = np.mean(ens_analysis, axis=0)
    ens_centered = ens_analysis - ens_mean_final
    ens_analysis = ens_mean_final + 1.02 * ens_centered
    
    # Step 5: Final safety checks
    ens_analysis = np.where(np.isfinite(ens_analysis), ens_analysis, 0.0)
    ens_analysis = np.clip(ens_analysis, -100.0, 100.0)
    
    # Time check
    if time.time() > deadline:
        return ens_analysis.astype(ensemble.dtype)
    
    return ens_analysis.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
