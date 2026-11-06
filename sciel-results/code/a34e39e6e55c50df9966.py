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


def _ensemble_kalman_filter(ensemble, obs_vec, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with optimal gain (regression-based) and adaptive inflation.
    Uses minimum-image distance localization on the ring.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_vec = np.asarray(obs_vec, dtype=np.float64).flatten()
    
    # Ensemble mean
    ens_mean = np.mean(ensemble, axis=0)
    
    # Centered anomalies
    anomalies = ensemble - ens_mean  # (N, D)
    
    # Observed anomalies
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    
    # Compute P_obs = cov(observed) + R
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    R = observation_variance * np.eye(M)
    S = P_obs + R  # (M, M)
    
    # Cross-covariance P_xH^T
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)
    
    # Localization: minimum-image distance on ring
    d_idx = np.arange(D)
    loc_weights = np.ones((D, M))
    L = 2.5  # tighter localization
    for m_idx, obs_i in enumerate(observed_indices):
        dist = np.abs(d_idx - obs_i)
        dist = np.minimum(dist, D - dist)
        loc_weights[:, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
    
    PHt_loc = PHt * loc_weights  # (D, M)
    
    # Compute S_inv
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Innovation for each member
    ens_obs = ensemble[:, observed_indices]  # (N, M)
    innovations = obs_vec[None, :] - ens_obs  # (N, M)
    
    # Gain-like update: delta_x = PHt_loc @ S_inv @ innovation
    Si_innov = innovations @ S_inv.T  # (N, M)
    delta_x = Si_innov @ PHt_loc.T  # (N, D)
    
    updated_ensemble = ensemble + delta_x  # (N, D)
    
    # Adaptive inflation: continuous function of spread/noise ratio
    ens_spread_obs = np.std(ens_obs, axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    mean_spread = np.mean(ens_spread_obs)
    
    # Smooth inflation: target ratio ~1.2
    ratio = mean_spread / obs_noise_std if obs_noise_std > 0 else 1.0
    target_ratio = 1.2
    # Inflation factor smoothly adjusts to bring spread closer to target
    inflation = 1.0 + 0.15 * (target_ratio - ratio) / max(ratio, 0.1)
    inflation = np.clip(inflation, 0.95, 1.12)
    
    updated_mean = np.mean(updated_ensemble, axis=0)
    updated_ensemble = updated_mean + inflation * (updated_ensemble - updated_mean)
    
    # Safety: clip and handle NaNs
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble


def _particle_filter_update(ensemble, obs_vec, observed_indices, observation_variance, F=None):
    """
    Nonlinear particle filter update using importance weights and kernel density estimation.
    Handles non-Gaussian error distributions inherent in chaotic systems.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_vec = np.asarray(obs_vec, dtype=np.float64).flatten()
    
    ens_mean = np.mean(ensemble, axis=0)
    
    # Compute likelihood weights based on observation match
    ens_obs = ensemble[:, observed_indices]  # (N, M)
    
    # Innovation
    innovations = obs_vec[None, :] - ens_obs  # (N, M)
    
    # Gaussian likelihood (log-space for stability)
    R_std = np.sqrt(observation_variance)
    log_weights = -0.5 * np.sum((innovations / R_std) ** 2, axis=1)
    
    # Normalize weights
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= weights.sum()
    
    # Check effective sample size
    ess = 1.0 / np.sum(weights ** 2)
    
    # If ESS is too low, use resampling; otherwise use weighted update
    if ess < N * 0.5:
        # Systematic resampling
        r = np.random.uniform(0, 1.0 / N, 1)
        cum_weights = np.cumsum(weights)
        cum_weights[-1] = 1.0
        indices = np.searchsorted(cum_weights, r + np.arange(N) / N)
        indices = np.clip(indices, 0, N - 1)
        resampled = ensemble[indices]
        
        # After resampling, apply a small Gaussian perturbation to avoid degeneracy
        # Use ensemble spread to set perturbation scale
        ens_spread = np.std(ensemble, axis=0)
        perturbation_scale = 0.1 * ens_spread
        resampled = resampled + np.random.randn(N, D) * perturbation_scale
    else:
        # Weighted kernel density estimate for posterior mean
        # Use Gaussian kernel with bandwidth based on ensemble spread
        bandwidth = 0.5 * np.std(ensemble, axis=0)
        bandwidth = np.maximum(bandwidth, 0.01)
        
        # Compute weighted posterior mean
        # For each state variable, compute weighted average with kernel smoothing
        # This is expensive, so use a simplified approach: weighted mean + correction
        weighted_mean = np.sum(weights[:, None] * ensemble, axis=0)
        
        # Compute weighted covariance for correction
        anomalies = ensemble - weighted_mean
        cov_weighted = (weights[:, None] * anomalies).T @ anomalies / (1.0 - np.sum(weights**2))
        
        # Simple correction: pull toward observation using localized gain
        obs_anomalies = anomalies[:, observed_indices]
        P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
        R = observation_variance * np.eye(M)
        S = P_obs + R
        PHt = (anomalies.T @ obs_anomalies) / (N - 1)
        
        # Localization
        d_idx = np.arange(D)
        loc_weights = np.ones((D, M))
        L = 2.5
        for m_idx, obs_i in enumerate(observed_indices):
            dist = np.abs(d_idx - obs_i)
            dist = np.minimum(dist, D - dist)
            loc_weights[:, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
        
        PHt_loc = PHt * loc_weights
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        # Gain correction
        gain = PHt_loc @ S_inv  # (D, M)
        innovation_mean = obs_vec - np.mean(ensemble[:, observed_indices], axis=0)
        correction = gain @ innovation_mean
        
        # Blend: 70% weighted mean, 30% EKF correction
        resampled = ensemble.copy()
        for i in range(N):
            resampled[i] = 0.7 * weighted_mean + 0.3 * (ensemble[i] + correction)
        
        # Re-center to maintain ensemble spread
        resampled_mean = np.mean(resampled, axis=0)
        original_spread = np.std(ensemble, axis=0)
        new_spread = np.std(resampled, axis=0)
        if np.any(new_spread > 0):
            scale = original_spread / np.maximum(new_spread, 0.01)
            scale = np.clip(scale, 0.8, 1.2)
            resampled = resampled_mean + scale * (resampled - resampled_mean)
    
    # Final safety
    resampled = np.nan_to_num(resampled, nan=0.0, posinf=100.0, neginf=-100.0)
    resampled = np.clip(resampled, -100.0, 100.0)
    
    return resampled


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
    deadline = start_time + 115.0  # safety margin
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations, dtype=np.float64)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr.flatten()
    
    # Primary: EKF with tight localization and smooth adaptive inflation
    result = _ensemble_kalman_filter(
        ensemble, obs_vec, observed_indices, observation_variance
    )
    
    # Deadline guard
    if time.time() > deadline:
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100.0, 100.0)
        return result
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
