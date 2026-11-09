"""Lorenz-96 ensemble assimilation with stochastic EnKF, inflation, and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter analysis update for Lorenz-96.
    Uses adaptive multiplicative inflation and Gaussian localization.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=np.int64)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # --- Compute ensemble mean and anomalies ---
    ens_mean = np.mean(ens, axis=0)
    ens_anom = ens - ens_mean  # (n_members, n_vars)
    
    # --- Adaptive Multiplicative Inflation ---
    # Innovation-based adaptive inflation with robust statistics
    innovation = obs - ens_mean[obs_idx]
    P_yy = np.mean(ens_anom[:, obs_idx] ** 2, axis=0)
    expected_innov_var = np.mean(P_yy + R_val)
    
    # Use median absolute deviation for robustness
    med_innov = np.median(innovation)
    mad = np.median(np.abs(innovation - med_innov))
    robust_innov_var = (1.4826 * mad) ** 2 if mad > 1e-10 else np.mean(innovation ** 2)
    
    if expected_innov_var > 1e-10:
        # Blend raw and robust innovation variance
        combined_innov_var = 0.5 * np.mean(innovation ** 2) + 0.5 * robust_innov_var
        ratio_innov = combined_innov_var / expected_innov_var
        inflation_adaptive = np.clip(ratio_innov, 0.97, 1.15)
    else:
        inflation_adaptive = 1.0
    
    # Spread-based component with tighter bounds
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-12))
    target_spread = np.sqrt(R_val) * 1.1
    spread_ratio = target_spread / np.maximum(current_spread, 1e-12)
    inflation_spread = np.clip(np.mean(spread_ratio), 0.99, 1.08)
    
    # Blend: primarily innovation-based with moderate spread correction
    inflation_factor = 0.6 * inflation_adaptive + 0.4 * inflation_spread
    inflation_factor = np.clip(inflation_factor, 1.0, 1.10)
    
    # Apply inflation
    ens = ens_mean + inflation_factor * ens_anom
    ens_mean = np.mean(ens, axis=0)
    ens_anom = ens - ens_mean
    
    # --- Localization weights ---
    sigma = 2.2  # Tighter localization for better Lorenz-96 correlation structure
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # (n_vars, n_obs)
    # Ensure observed locations have weight 1
    for i, oi in enumerate(obs_idx):
        loc_weights[oi, i] = 1.0
    
    # --- Ensemble Kalman Filter Update ---
    # Anomalies at observed locations
    A_obs = ens_anom[:, obs_idx]  # (n_members, n_obs)
    
    # Cross-covariance between all variables and observations
    P_xo = (ens_anom.T @ A_obs) / (n_members - 1.0)  # (n_vars, n_obs)
    
    # Apply localization
    P_xo_loc = P_xo * loc_weights
    
    # Observation error variance with localization
    P_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # (n_obs, n_obs)
    
    # Apply localization to observation covariance as well
    obs_dist = np.abs(obs_idx[:, None] - obs_idx[None, :])
    obs_dist = np.minimum(obs_dist, n_vars - obs_dist)
    obs_loc = np.exp(-0.5 * (obs_dist / sigma) ** 2)
    P_oo_loc = P_oo * obs_loc
    
    S = P_oo_loc + R_val * np.eye(n_obs)
    
    # Cholesky of S with adaptive jitter
    jitter = 1e-10
    for _ in range(10):
        try:
            L = np.linalg.cholesky(S + jitter * np.eye(n_obs))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    else:
        L = np.linalg.cholesky(S + 1e-6 * np.eye(n_obs))
    
    # Compute S_inv using Cholesky
    L_inv = np.linalg.inv(L)
    S_inv = L_inv.T @ L_inv  # (n_obs, n_obs)
    
    # Gain matrix
    K = P_xo_loc @ S_inv  # (n_vars, n_obs)
    
    # Update each member: x_a_i = x_f_i + K @ (obs - x_f_i[obs_idx])
    member_innovations = obs[None, :] - (ens_mean[obs_idx][None, :] + A_obs)  # (n_members, n_obs)
    
    # Update
    ens = ens + (K @ member_innovations.T).T  # (n_members, n_vars)
    
    # --- Post-processing: ensure finite and bounded ---
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
    
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
