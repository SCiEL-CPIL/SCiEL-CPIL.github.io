"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system (vectorized)."""
    x_ip1 = np.roll(x, -1)
    x_im1 = np.roll(x, 1)
    x_im2 = np.roll(x, 2)
    return (x_ip1 - x_im2) * x_im1 - x + F


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _compute_localization_matrix(D, observed_indices, L):
    """
    Compute localization weights matrix of shape (D, M).
    D = state dimension, M = number of observations.
    Uses minimum-image distance on the ring.
    """
    M = len(observed_indices)
    weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            weights[d, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
    return weights


def _enkf_update(ensemble, obs_vec, obs_idx, obs_var, D, M, loc_width):
    """Standard EnKF update with localization."""
    N = ensemble.shape[0]
    
    loc_mat = _compute_localization_matrix(D, obs_idx, loc_width)
    
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    obs_anomalies = anomalies[:, obs_idx]
    
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
    R = obs_var * np.eye(M)
    S = P_obs + R
    
    # Eigenvalue-based regularization
    eigvals = np.linalg.eigvalsh(S)
    min_eig = np.min(eigvals)
    if min_eig < 1e-6:
        S += (1e-6 - min_eig) * np.eye(M)
    else:
        S += 1e-8 * np.eye(M)
    
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)
    PHt_loc = PHt * loc_mat
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = PHt_loc @ S_inv
    
    innovations = obs_vec[None, :] - ensemble[:, obs_idx]
    deltas = innovations @ K.T
    updated = ensemble + deltas
    
    return updated


def _ensrf_update(ensemble, obs_vec, obs_idx, obs_var, D, M, loc_width):
    """
    Ensemble Square Root Filter (EnSRF) update using regression.
    Maps ensemble perturbations in state space to observation space via least-squares,
    then uses the inverse mapping to update unobserved components.
    """
    N = ensemble.shape[0]
    
    loc_mat = _compute_localization_matrix(D, obs_idx, loc_width)
    
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)
    
    # Observation-space anomalies
    obs_anoms = anomalies[:, obs_idx]  # (N, M)
    
    # For EnSRF, we want to find a linear map A such that:
    # obs_anoms ≈ anomalies @ A  (in least squares sense)
    # Then for the update, we project the innovation back.
    
    # Actually, the standard EnSRF approach:
    # 1. Compute the observation-space innovation: y - Hx_mean
    # 2. Find the regression coefficients that best predict observations from state
    # 3. Use those to update
    
    # Regression: obs_anoms = anomalies @ W + noise
    # W is (D, M), solve: W = pinv(anomalies) @ obs_anoms
    # But we want the gain: how to update state from observation innovation
    
    # The EnSRF gain is: K = (H P H^T + R)^-1 H P
    # where P is the ensemble covariance
    # This is actually the same as EnKF but computed differently.
    
    # Alternative EnSRF: use the fact that with N members, we can do
    # a direct square-root update without explicitly forming the covariance.
    
    # Let's use the regression-based approach:
    # Fit: obs_anoms ≈ anomalies @ C  where C is (D, M)
    # C = (anomalies^T anomalies)^-1 anomalies^T obs_anoms
    # But this is just the cross-covariance divided by state covariance.
    
    # The proper EnSRF: 
    # innovation = obs_vec - ens_mean[obs_idx]  (M,)
    # The update to the mean is: delta_mean = P H^T (H P H^T + R)^-1 innovation
    # For each member: x_i_new = x_mean_new + (x_i - x_mean_old) @ K_state
    # where K_state maps state anomalies to updated state anomalies
    
    # Let me implement the standard EnSRF algorithm:
    # 1. Compute P = (1/(N-1)) * A^T A where A = anomalies (N, D)
    # 2. H P H^T = (1/(N-1)) * obs_anoms^T obs_anoms
    # 3. S = H P H^T + R
    # 4. K = P H^T S^-1 = (1/(N-1)) A^T obs_anoms S^-1
    # 5. Update mean: x_mean_new = x_mean + K @ innovation
    # 6. Update anomalies: A_new = A @ (I - K H) ... this gets complicated
    
    # Simpler approach that captures nonlinearity better:
    # Use local linear regression per observation with localization
    
    # For each state component d, fit a linear model:
    # x_d = sum_m beta_{d,m} * y_m + intercept
    # using only nearby observations (localization)
    
    # This is essentially what the EnKF does, but let's try a different angle:
    # Use the ensemble to estimate the conditional distribution P(x_d | y_obs)
    
    # Practical EnSRF implementation:
    # The key insight is that EnSRF avoids sampling error by working
    # directly with the square root of the covariance.
    
    # For our purposes, let's implement a variant that uses
    # the ensemble to build a local regression model.
    
    # Build regression: for each state variable d, regress on observed variables
    # with localization weights
    
    # Actually, the most effective "nonlinear" improvement within the EnKF family
    # is to use a nonlinear regression: fit a quadratic or use kernel methods.
    # But with only 16 members, we need to be careful.
    
    # Let's use the standard EnKF but with a refined gain that accounts
    # for the nonlinear structure by using a two-step process:
    # 1. Linear EnKF update
    # 2. Refine using the ensemble's nonlinear structure
    
    # For now, implement the standard EnKF with better localization
    # and adaptive inflation, which is what the inspirations suggest works best.
    
    # Compute the standard EnKF gain
    P_obs = (obs_anoms.T @ obs_anoms) / (N - 1)
    R = obs_var * np.eye(M)
    S = P_obs + R
    
    eigvals = np.linalg.eigvalsh(S)
    min_eig = np.min(eigvals)
    if min_eig < 1e-6:
        S += (1e-6 - min_eig) * np.eye(M)
    else:
        S += 1e-8 * np.eye(M)
    
    PHt = (anomalies.T @ obs_anoms) / (N - 1)
    PHt_loc = PHt * loc_mat
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = PHt_loc @ S_inv
    
    # Update
    innovations = obs_vec[None, :] - ensemble[:, obs_idx]
    deltas = innovations @ K.T
    updated = ensemble + deltas
    
    return updated


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for partial-observation Lorenz-96.
    
    Parameters:
    -----------
    ensemble : (16, 20) array - ensemble of 16 members, 20 state variables
    observations : (10,) or (16, 10) array - noisy observations of even-indexed coordinates
    observed_indices : array of 10 indices - the even indices [0, 2, 4, ..., 18]
    observation_variance : float - variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) array - updated ensemble
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_idx = np.asarray(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr.flatten()
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble = np.clip(ensemble, -100.0, 100.0)
    obs_vec = np.nan_to_num(obs_vec, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # First pass: EnKF with narrower localization
    L_loc_1 = 2.5
    updated_1 = _enkf_update(ensemble, obs_vec, obs_idx, observation_variance, D, M, L_loc_1)
    
    # Check time
    if time.time() > deadline - 2.0:
        updated_ensemble = updated_1
        updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
        updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
        return updated_ensemble
    
    # Second pass: EnKF with wider localization on the first-pass result
    # This smooths out inconsistencies from the first update
    L_loc_2 = 4.0
    updated_2 = _enkf_update(updated_1, obs_vec, obs_idx, observation_variance, D, M, L_loc_2)
    
    # Check time
    if time.time() > deadline - 1.0:
        updated_ensemble = updated_2
        updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
        updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
        return updated_ensemble
    
    # Adaptive inflation based on innovation statistics
    obs_spread_pre = np.std(ensemble[:, obs_idx], axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    mean_ens_spread = np.mean(obs_spread_pre)
    
    # Smooth adaptive inflation
    spread_ratio = mean_ens_spread / obs_noise_std if obs_noise_std > 0 else 1.0
    
    if spread_ratio < 1.0:
        inflation = 1.0 + 0.15 * (1.0 - spread_ratio)
    elif spread_ratio > 1.5:
        inflation = 1.0 - 0.05 * (spread_ratio - 1.5)
    else:
        inflation = 1.0
    
    inflation = max(0.95, min(1.15, inflation))
    
    if abs(inflation - 1.0) > 1e-6:
        ens_mean_2 = np.mean(updated_2, axis=0)
        updated_2 = ens_mean_2 + inflation * (updated_2 - ens_mean_2)
    
    updated_ensemble = updated_2
    
    # Final safety checks
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble

# EVOLVE-BLOCK-END
