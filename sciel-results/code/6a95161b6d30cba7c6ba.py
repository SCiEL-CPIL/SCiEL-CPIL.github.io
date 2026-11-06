"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None
_DEADLINE_FRACTION = 0.90

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


def _enskf_update(ensemble, obs_vec, observed_indices, observation_variance,
                  localization_length=3.0, rel_spread=0.1, beta=1.0):
    """
    Ensemble Kalman Filter update with multiplicative spread preservation.
    Uses the standard EnKF with localization and adaptive inflation.
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
    
    # Multiplicative spread preservation
    new_mean = np.mean(updated_ensemble, axis=0)
    new_anomalies = updated_ensemble - new_mean
    
    # Compute the ratio of ensemble variance before and after update
    old_var = np.sum(anomalies ** 2, axis=0) / (N - 1)  # (D,)
    new_var = np.sum(new_anomalies ** 2, axis=0) / (N - 1)  # (D,)
    
    # Preserve spread: scale anomalies so that variance is maintained
    # with a small inflation factor to account for model error
    scale = np.ones(D)
    for d in range(D):
        if new_var[d] > 1e-10:
            target_var = old_var[d] * (1.0 + rel_spread)
            scale[d] = np.sqrt(target_var / new_var[d])
        else:
            scale[d] = 1.0
    
    # Limit the scale to avoid excessive inflation
    scale = np.clip(scale, 0.5, 3.0)
    
    updated_ensemble = new_mean + scale * new_anomalies
    
    return updated_ensemble


def _dual_ensemble_update(ensemble, obs_vec, observed_indices, observation_variance):
    """
    A more sophisticated update that combines multiple strategies.
    Uses ensemble transform ideas to better handle non-Gaussianity.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    # Compute innovation statistics
    Hx = ensemble[:, observed_indices]
    innovations = obs_vec[None, :] - Hx
    
    # Standard EnKF update
    ekf_updated = _enskf_update(ensemble, obs_vec, observed_indices, 
                                observation_variance, localization_length=3.0)
    
    # Compute how well the ensemble predicted the observations
    obs_anomalies = anomalies[:, observed_indices]
    P_oo = (obs_anomalies.T @ obs_anomalies) / (N - 1)
    R = observation_variance * np.eye(M)
    S = P_oo + R
    
    expected_innov_var = np.diag(S)
    actual_innov_var = np.var(innovations, axis=0)
    
    # Innovation ratio
    innov_ratio = np.mean(actual_innov_var / (expected_innov_var + 1e-10))
    
    # If the ensemble is under-dispersed, apply stronger inflation
    if innov_ratio > 1.5:
        # Strong inflation
        new_mean = np.mean(ekf_updated, axis=0)
        new_anoms = ekf_updated - new_mean
        inflation = 1.0 + 0.3 * (innov_ratio - 1.0)
        inflation = min(inflation, 2.0)
        ekf_updated = new_mean + inflation * new_anoms
    
    # Blend with a simple observation-constrained mean
    # For observed coordinates, pull the mean toward the observation
    obs_mean = np.mean(obs_vec)
    for i, idx in enumerate(observed_indices):
        # Gentle pull toward observation
        ekf_updated[:, idx] = 0.95 * ekf_updated[:, idx] + 0.05 * obs_vec[i]
    
    return ekf_updated


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
    
    # Use the dual ensemble update which combines EnKF with adaptive inflation
    updated_ensemble = _dual_ensemble_update(ensemble, obs_vec, observed_indices, 
                                             observation_variance)
    
    # Safety checks
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble

# EVOLVE-BLOCK-END
