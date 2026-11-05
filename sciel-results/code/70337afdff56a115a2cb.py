"""
Local Ensemble Kalman Filter (LEKF) with inflation and localization for Lorenz-96 assimilation.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x has shape (N, 20)."""
    x_1 = np.roll(x, 1, axis=1)
    x_m1 = np.roll(x, -1, axis=1)
    x_m2 = np.roll(x, 2, axis=1)
    return (x_1 - x_m2) * x_m1 - x + F

def _rk4_step(x, F, dt=0.01, n_steps=5):
    """Propagate ensemble forward using RK4 for n_steps steps of size dt."""
    x = x.copy()
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x

def _localization_weights(n_obs, n_state, decay=0.05):
    """
    Compute localization weights for observation influence on state variables.
    Uses exponential decay based on distance in the periodic 20-variable space.
    """
    weights = np.zeros((n_obs, n_state))
    for i, obs_idx in enumerate(range(n_obs)):
        for j in range(n_state):
            # Periodic distance
            dist = min(abs(j - obs_idx), n_state - abs(j - obs_idx))
            weights[i, j] = np.exp(-decay * dist)
    return weights

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Kalman Filter update for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (8,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices [0, 2, 4, ..., 18]
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin before 120s limit
    
    n_members, n_state = ensemble.shape
    n_obs = len(observed_indices)
    
    # Estimate forcing F from ensemble mean dynamics
    # Use the fact that in steady state, the mean should approximately satisfy the dynamics
    # F estimate: average of (x + (x_m2 - x_1)*x_m1) over ensemble
    x_mean = np.mean(ensemble, axis=0)
    x_1 = np.roll(x_mean, 1)
    x_m1 = np.roll(x_mean, -1)
    x_m2 = np.roll(x_mean, 2)
    F_est = np.mean(x_mean + (x_m2 - x_1) * x_m1)
    # Clamp F to reasonable range
    F_est = np.clip(F_est, 5.0, 12.0)
    
    # Propagate ensemble forward
    ensemble_prop = _rk4_step(ensemble, F_est, dt=0.01, n_steps=5)
    
    # Compute background mean and perturbation
    bg_mean = np.mean(ensemble_prop, axis=0)
    bg_pert = ensemble_prop - bg_mean
    
    # Compute background covariance at observed indices
    # C_bg = (1/(n-1)) * X_pert @ X_pert.T for observed indices
    obs_pert = bg_pert[:, observed_indices]
    C_bg = (obs_pert @ obs_pert.T) / (n_members - 1.0)
    
    # Add small regularization for numerical stability
    C_bg += 1e-6 * np.eye(n_obs)
    
    # Observation error covariance
    R = observation_variance * np.eye(n_obs)
    
    # Innovation: observations - predicted observations
    innovation = observations - bg_mean[observed_indices]
    
    # Compute Kalman gain for observed variables
    # K = C_bg @ (C_bg + R)^{-1}
    S = C_bg + R
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K_obs = C_bg @ S_inv  # Shape: (n_obs, n_obs)
    
    # Apply update to observed variables
    update_obs = K_obs @ innovation  # Shape: (n_obs,)
    
    # Compute localization weights for unobserved variables
    # For each state variable, compute its correlation-like weight with observed variables
    # Using exponential decay in periodic distance
    decay = 0.08
    loc_weights = np.zeros((n_state, n_obs))
    for j in range(n_state):
        for i, obs_idx in enumerate(observed_indices):
            dist = min(abs(j - obs_idx), n_state - abs(j - obs_idx))
            loc_weights[j, i] = np.exp(-decay * dist)
    
    # Full update for all state variables
    # For observed variables: use full Kalman update
    # For unobserved variables: use localized gain based on correlation with observed
    
    updated = ensemble_prop.copy()
    
    # Update observed variables directly
    updated[:, observed_indices] += update_obs[np.newaxis, :]
    
    # Update unobserved variables using localized approach
    # For each unobserved variable j, compute its estimated update as a weighted sum of observed updates
    # Weight by the ratio of cross-covariance to observation covariance
    unobs_indices = [i for i in range(n_state) if i not in observed_indices]
    
    for j in unobs_indices:
        # Cross-covariance between variable j and observed variables
        # C_j,obs = (1/(n-1)) * x_j_pert @ obs_pert.T
        x_j_pert = bg_pert[:, j]
        C_j_obs = (x_j_pert @ obs_pert.T) / (n_members - 1.0)  # Shape: (1, n_obs)
        
        # Gain for variable j: C_j_obs @ (C_bg + R)^{-1}
        K_j = C_j_obs @ S_inv  # Shape: (1, n_obs)
        
        # Apply localization
        K_j_loc = K_j * loc_weights[j, :][np.newaxis, :]
        
        # Update
        update_j = K_j_loc @ innovation  # Shape: (1,)
        updated[:, j] += update_j[0]
    
    # Apply multiplicative inflation to prevent filter collapse
    # Compute ensemble spread and adjust
    new_mean = np.mean(updated, axis=0)
    new_pert = updated - new_mean
    spread2 = np.mean(new_pert ** 2, axis=0)
    
    # Target spread based on background spread
    bg_spread2 = np.mean(bg_pert ** 2, axis=0)
    
    # Inflation factor: adjust spread to be at least as large as background
    # Use a gentle inflation
    inflation_factor = np.ones(n_state)
    for j in range(n_state):
        if spread2[j] < 0.5 * bg_spread2[j]:
            inflation_factor[j] = np.sqrt(bg_spread2[j] / (spread2[j] + 1e-10))
            inflation_factor[j] = min(inflation_factor[j], 1.5)  # Cap inflation
    
    # Apply inflation
    updated = new_mean[np.newaxis, :] + new_pert * inflation_factor[np.newaxis, :]
    
    # Ensure finiteness and bounds
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    updated = np.clip(updated, -100.0, 100.0)
    
    # Deadline check
    if time.time() > deadline:
        return updated
    
    return updated

# EVOLVE-BLOCK-END
