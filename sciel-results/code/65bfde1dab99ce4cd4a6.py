"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Global deadline tracking
_start_time = None
_DEADLINE_FRACTION = 0.90  # Use 90% of the 150s budget

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system using vectorized operations."""
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


def _lorenz96_propagate(x, F, dt, nsteps):
    """Propagate Lorenz-96 state forward."""
    for _ in range(nsteps):
        x = _rk4_step(x, F, dt)
    return x


def _gaussian_localization(dist, L):
    """Gaussian localization function."""
    return np.exp(-0.5 * (dist / L) ** 2)


def _compute_localization_weights(D, observed_indices, localization_length):
    """
    Pre-compute localization weights for cross-covariance.
    Returns a (D, M) matrix of weights.
    """
    M = len(observed_indices)
    weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            weights[d, m_idx] = _gaussian_localization(dist, localization_length)
    return weights


def _ensmith_update(ensemble, obs_vec, observed_indices, observation_variance,
                    localization_length=2.0):
    """
    Ensemble Smoother with Multiplicative Inflation (ESMF) style update.
    Uses a single-step assimilation approach with adaptive inflation.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)
    
    # Observation anomalies
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    
    # Covariances
    P_oo = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    R = observation_variance * np.eye(M)
    S = P_oo + R  # (M, M)
    
    P_ox = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)
    
    # Localization
    loc_weights = _compute_localization_weights(D, observed_indices, localization_length)
    P_ox_loc = P_ox * loc_weights
    
    # Kalman Gain
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = P_ox_loc @ S_inv  # (D, M)
    
    # Innovation
    Hx = ensemble[:, observed_indices]  # (N, M)
    innovations = obs_vec[None, :] - Hx  # (N, M)
    
    # Update
    delta_ensemble = innovations @ K.T  # (N, D)
    updated_ensemble = ensemble + delta_ensemble
    
    # Adaptive multiplicative inflation
    new_mean = np.mean(updated_ensemble, axis=0)
    new_anomalies = updated_ensemble - new_mean
    
    # Compute innovation variance ratio
    expected_innov_var = np.trace(S)
    actual_innov_var = np.mean(np.sum(innovations ** 2, axis=1))
    
    if expected_innov_var > 1e-10:
        ratio = actual_innov_var / expected_innov_var
        inflation_factor = 1.0 + 0.5 * max(0, ratio - 1.0)
        inflation_factor = min(inflation_factor, 2.0)
        
        if inflation_factor > 1.0:
            updated_ensemble = new_mean + inflation_factor * new_anomalies
    
    return updated_ensemble


def _particle_filter_update(ensemble, obs_vec, observed_indices, observation_variance,
                            localization_length=2.0):
    """
    Sequential Monte Carlo / Particle Filter style update.
    Uses importance weighting based on observation likelihood.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    # Compute log-likelihood weights
    # For each member, compute the innovation
    Hx = ensemble[:, observed_indices]  # (N, M)
    innovations = obs_vec[None, :] - Hx  # (N, M)
    
    # Approximate covariance of observations
    obs_anomalies = anomalies[:, observed_indices]
    P_oo = (obs_anomalies.T @ obs_anomalies) / (N - 1)
    R = observation_variance * np.eye(M)
    S = P_oo + R
    
    # Log-likelihood: -0.5 * v^T S^{-1} v
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    log_weights = np.zeros(N)
    for i in range(N):
        v = innovations[i]
        log_weights[i] = -0.5 * v @ S_inv @ v
    
    # Normalize weights (log-sum-exp for stability)
    max_log_w = np.max(log_weights)
    weights = np.exp(log_weights - max_log_w)
    weights /= np.sum(weights)
    
    # Resampling: systematic resampling
    cum_weights = np.cumsum(weights)
    cum_weights[-1] = 1.0  # Ensure exact sum
    
    # Systematic resampling
    u = np.random.uniform(0, 1.0 / N)
    indices = np.zeros(N, dtype=int)
    for i in range(N):
        u_i = u + i / N
        indices[i] = np.searchsorted(cum_weights, u_i)
    
    # Resampled ensemble
    resampled = ensemble[indices]
    
    # Add small perturbation to avoid degeneracy
    noise_scale = 0.1 * np.std(ensemble, axis=0)
    resampled = resampled + np.random.normal(0, 1, size=(N, D)) * noise_scale
    
    # Final mean-based adjustment toward observations
    new_mean = np.mean(resampled, axis=0)
    # Blend with observation-informed mean
    obs_informed = ens_mean.copy()
    obs_informed[observed_indices] = 0.7 * obs_informed[observed_indices] + 0.3 * obs_vec
    blended_mean = 0.5 * new_mean + 0.5 * obs_informed
    
    # Re-center
    resampled = blended_mean + (resampled - new_mean)
    
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
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 150 * _DEADLINE_FRACTION:
        # Deadline exceeded, return current ensemble as-is
        return np.clip(ensemble, -100.0, 100.0)
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Ensure obs_vec is 1D
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr
    
    # Strategy: Use a hybrid approach combining EKF with adaptive inflation
    # and a particle filter component for better handling of non-Gaussianity
    
    # First, compute the EKF update
    ekf_updated = _ensmith_update(ensemble, obs_vec, observed_indices, 
                                   observation_variance, localization_length=2.0)
    
    # Compute ensemble spread
    ens_mean = np.mean(ensemble, axis=0)
    ens_std = np.std(ensemble, axis=0)
    
    # Compute innovation statistics
    Hx = ensemble[:, observed_indices]
    innovations = obs_vec[None, :] - Hx
    innov_mean = np.mean(innovations, axis=0)
    innov_std = np.std(innovations, axis=0)
    
    # If innovations are large relative to expected, the ensemble may be under-dispersed
    expected_innov_std = np.sqrt(np.diag((np.cov(ensemble[:, observed_indices].T) + 
                                           observation_variance * np.eye(M))))
    
    # Adaptive strategy: blend EKF with a simpler observation blending
    # based on how well the ensemble predicts the observations
    
    pred_error = np.mean(np.abs(innov_mean)) / (np.mean(expected_innov_std) + 1e-10)
    
    if pred_error > 2.0:
        # Ensemble is significantly under-dispersed; use more aggressive inflation
        # and stronger observation blending
        new_mean = np.mean(ekf_updated, axis=0)
        anomalies = ekf_updated - new_mean
        
        # Strong inflation
        inflation = 1.5 + 0.5 * min(pred_error - 2.0, 2.0)
        inflated = new_mean + inflation * anomalies
        
        # Additional observation blending for observed coordinates
        obs_blend = inflated.copy()
        for i, idx in enumerate(observed_indices):
            obs_blend[:, idx] = 0.6 * obs_blend[:, idx] + 0.4 * obs_vec[i]
        
        updated_ensemble = obs_blend
    else:
        # Normal case: use EKF result with mild adaptive inflation
        new_mean = np.mean(ekf_updated, axis=0)
        anomalies = ekf_updated - new_mean
        
        # Mild inflation based on innovation ratio
        if np.mean(expected_innov_std) > 1e-10:
            ratio = np.mean(innov_std) / np.mean(expected_innov_std)
            inflation = 1.0 + 0.3 * max(0, ratio - 1.0)
            inflation = min(inflation, 1.5)
        else:
            inflation = 1.0
        
        updated_ensemble = new_mean + inflation * anomalies
    
    # Safety checks
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble

# EVOLVE-BLOCK-END
