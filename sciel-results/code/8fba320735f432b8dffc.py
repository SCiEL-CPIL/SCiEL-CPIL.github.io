"""Lorenz-96 ensemble assimilation with simultaneous EnKF, adaptive inflation, and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) analysis update for Lorenz-96.
    Uses simultaneous update with adaptive multiplicative inflation and Gaussian localization.
    
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
    deadline = start_time + 110.0
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices)
    R_val = observation_variance
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # Ensure finite inputs
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # --- Adaptive Multiplicative Inflation (tanh-based, smooth) ---
    ens_mean = np.mean(ens, axis=0)
    ens_centered = ens - ens_mean
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    
    # Target: maintain spread at ~1.3x sqrt(R)
    target_spread = np.sqrt(R_val) * 1.3
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-12))
    
    ratio = target_spread / np.maximum(current_spread, 1e-12)
    inflation_correction = np.tanh(0.5 * (np.mean(ratio) - 1.0))
    inflation = 1.0 + 0.3 * inflation_correction
    inflation = np.clip(inflation, 1.0, 1.12)
    
    # Apply inflation
    ens = ens_mean + inflation * ens_centered
    ens_mean = np.mean(ens, axis=0)
    
    # --- Simultaneous EnKF Update ---
    A = ens - ens_mean  # (n_members, n_vars)
    A_obs = A[:, obs_idx]  # (n_members, n_obs)
    
    # Cross-covariance: (n_vars, n_obs)
    C_xo = (A.T @ A_obs) / (n_members - 1.0)
    
    # Observation covariance: (n_obs, n_obs)
    C_oo = (A_obs.T @ A_obs) / (n_members - 1.0)
    
    # Gaussian localization
    sigma = 2.5
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # (n_vars, n_obs)
    
    # Localized cross-covariance
    C_xo_loc = C_xo * loc_weights
    
    # Innovation
    innovation = obs - ens_mean[obs_idx]  # (n_obs,)
    
    # Gain computation with regularization
    reg = 1e-6
    S = C_oo + R_val * np.eye(n_obs) + reg * np.eye(n_obs)
    
    try:
        K = np.linalg.solve(S, C_xo_loc.T).T  # (n_vars, n_obs)
    except np.linalg.LinAlgError:
        K = C_xo_loc @ np.linalg.pinv(S)
    
    # Update mean
    delta_mean = K @ innovation
    ens_mean = ens_mean + delta_mean
    
    # Update anomalies
    A_obs_Kt = A_obs @ K.T  # (n_members, n_vars)
    A_new = A - A_obs_Kt
    
    # Reconstruct ensemble
    ens = ens_mean[None, :] + A_new
    
    # --- Final spread check and gentle correction ---
    ens_mean_final = np.mean(ens, axis=0)
    ens_centered_final = ens - ens_mean_final
    final_var_obs = np.var(ens[:, obs_idx], axis=0)
    final_spread = np.sqrt(np.maximum(final_var_obs, 1e-12))
    target_final = np.sqrt(R_val) * 1.1
    spread_ratio = target_final / np.maximum(final_spread, 1e-12)
    
    # Only apply if spread is too low (deflation)
    if np.mean(spread_ratio) > 1.05:
        correction = np.clip(1.0 + 0.2 * np.tanh(0.3 * (np.mean(spread_ratio) - 1.0)), 1.0, 1.08)
        ens = ens_mean_final + correction * ens_centered_final
    
    # --- Post-processing ---
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
    
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
