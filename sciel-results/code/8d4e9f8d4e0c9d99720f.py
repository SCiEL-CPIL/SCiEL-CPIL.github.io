"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE_SECONDS = 115.0
_START_TIME = [time.time()]


def _elapsed():
    """Return elapsed time since first call."""
    return time.time() - _START_TIME[0]


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


def _ensemble_kalman_filter(ensemble, obs_vec, observed_indices, observation_variance,
                            localization_length=2.5, inflation_factor=1.03,
                            max_inflation=1.3):
    """
    Vectorized Ensemble Kalman Filter update with Gaussian localization and adaptive inflation.
    
    Parameters:
    -----------
    ensemble : (N, D) array - ensemble of states
    obs_vec : (M,) array - single observation vector
    observed_indices : array of indices that are observed
    observation_variance : float - variance of observation noise
    localization_length : float - length scale for Gaussian localization
    inflation_factor : float - base multiplicative inflation factor
    max_inflation : float - maximum allowed inflation
    
    Returns:
    --------
    updated_ensemble : (N, D) array
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Ensure obs_vec is 1D
    obs_vec = np.asarray(obs_vec).flatten()
    
    # Compute ensemble mean
    ens_mean = np.mean(ensemble, axis=0)
    
    # Centered ensemble (anomalies)
    anomalies = ensemble - ens_mean  # (N, D)
    
    # Observation operator: H selects observed components
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    
    # Compute observation-space covariance P_obs
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    
    # Add observation noise
    R = observation_variance * np.eye(M)
    S = P_obs + R  # Innovation covariance (M, M)
    
    # Compute cross-covariance PHt
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)
    
    # Apply localization to the cross-covariance
    loc_weights = _compute_localization_matrix(D, observed_indices, localization_length)
    PHt_loc = PHt * loc_weights  # (D, M)
    
    # Solve for S_inv using Cholesky for numerical stability
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Vectorized update:
    # innovations = obs_vec[None, :] - ensemble[:, observed_indices]  (N, M)
    # delta_x = innovations @ S_inv.T @ PHt_loc.T  (N, D)
    innovations = obs_vec[None, :] - ensemble[:, observed_indices]  # (N, M)
    # delta_x[n, d] = sum_m innovations[n,m] * S_inv[m,k] * PHt_loc[d,k]
    # = sum_k (sum_m innovations[n,m] * S_inv[m,k]) * PHt_loc[d,k]
    # = (innovations @ S_inv.T) @ PHt_loc.T
    weighted_innov = innovations @ S_inv.T  # (N, M)
    delta_x = weighted_innov @ PHt_loc.T  # (N, D)
    
    updated_ensemble = ensemble + delta_x
    
    # Adaptive inflation based on innovation statistics
    obs_mean_pre = np.mean(ensemble[:, observed_indices], axis=0)
    obs_spread_pre = np.std(ensemble[:, observed_indices], axis=0, ddof=1)
    
    obs_noise_std = np.sqrt(observation_variance)
    mean_ens_spread = np.mean(obs_spread_pre)
    
    # Compare ensemble spread to expected innovation std
    # Expected innovation std ~ sqrt(ens_spread^2 + obs_noise^2)
    expected_innov_std = np.sqrt(np.mean(obs_spread_pre**2) + observation_variance)
    actual_innov_std = np.std(innovations, axis=0)
    mean_actual_innov = np.mean(actual_innov_std)
    
    # If actual innovations are larger than expected, ensemble is under-dispersed
    if mean_ens_spread < obs_noise_std * 0.3:
        inflation = 1.08
    elif mean_actual_innov > expected_innov_std * 1.5:
        inflation = 1.05
    else:
        inflation = 1.0
    
    # Apply inflation about the ensemble mean
    if inflation > 1.0:
        updated_ensemble = ens_mean + inflation * (updated_ensemble - ens_mean)
    
    # Ensure finite values and clip to valid range
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble


def _compute_localization_matrix(D, observed_indices, L):
    """Compute localization matrix for cross-covariance.

    Returns (D, M) matrix of localization weights.
    """
    M = len(observed_indices)
    loc = np.ones((D, M))
    d_idx = np.arange(D)
    for m_idx, obs_i in enumerate(observed_indices):
        dist = np.abs(d_idx - obs_i)
        dist = np.minimum(dist, D - dist)
        loc[:, m_idx] = _gaussian_localization(dist, L)
    return loc


def _gaussian_localization(dist, L):
    """Gaussian localization function."""
    return np.exp(-0.5 * (dist / L) ** 2)


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
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr
    
    # Check deadline
    if _elapsed() > _DEADLINE_SECONDS - 5.0:
        # Return ensemble mean as fallback if we're out of time
        ens_mean = np.mean(ensemble, axis=0, keepdims=True)
        return np.tile(ens_mean, (N, 1))
    
    # Vectorized EKF with localization and inflation
    result = _ensemble_kalman_filter(
        ensemble, obs_vec, observed_indices, observation_variance,
        localization_length=2.5,
        inflation_factor=1.03,
        max_inflation=1.3
    )
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
