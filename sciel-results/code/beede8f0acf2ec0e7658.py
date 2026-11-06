"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START


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
    deadline = start_time + 115.0  # safety margin before 150s kill

    N, D = ensemble.shape
    M = len(observed_indices)

    # Handle observations shape
    obs_arr = np.asarray(observations, dtype=np.float64)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0].copy()
    else:
        obs_vec = obs_arr.flatten().copy()

    # Ensure observed_indices is a numpy array
    observed_indices = np.asarray(observed_indices, dtype=int)

    # --- Ensemble Kalman Filter with localization and adaptive inflation ---

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
    L = 2.0  # tighter localization for Lorenz-96 ring structure
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

    # --- Adaptive spread inflation ---
    ens_spread_obs = np.std(ens_obs, axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    mean_spread = np.mean(ens_spread_obs)

    # Smooth inflation: target ratio ~1.3 (slightly over-dispersed)
    ratio = mean_spread / obs_noise_std if obs_noise_std > 0 else 1.0
    target_ratio = 1.3
    inflation = 1.0 + 0.20 * (target_ratio - ratio) / max(ratio, 0.1)
    inflation = np.clip(inflation, 0.92, 1.15)

    updated_mean = np.mean(updated_ensemble, axis=0)
    updated_ensemble = updated_mean + inflation * (updated_ensemble - updated_mean)

    # --- Nonlinear correction via particle-weighted refinement ---
    # Use likelihood weights to identify which members are most consistent
    # with observations, then apply a soft blending step.
    ens_obs_updated = updated_ensemble[:, observed_indices]
    innovations_pf = obs_vec[None, :] - ens_obs_updated
    R_std = np.sqrt(observation_variance)
    log_weights = -0.5 * np.sum((innovations_pf / R_std) ** 2, axis=1)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= weights.sum()

    # Effective sample size
    ess = 1.0 / np.sum(weights ** 2)

    if ess > N * 0.3:
        # Weighted posterior mean estimate
        weighted_mean = np.sum(weights[:, None] * updated_ensemble, axis=0)
        # Blend: keep ensemble structure but shift toward weighted mean
        # This corrects for non-Gaussianity without collapsing spread
        blend = 0.25
        updated_mean2 = np.mean(updated_ensemble, axis=0)
        spread = np.std(updated_ensemble, axis=0)
        target_mean = (1 - blend) * updated_mean2 + blend * weighted_mean
        # Re-center to preserve spread
        updated_ensemble = target_mean + (updated_ensemble - updated_mean2)
    else:
        # Low ESS: resample to avoid degeneracy
        r = np.random.uniform(0, 1.0 / N, 1)
        cum_weights = np.cumsum(weights)
        cum_weights[-1] = 1.0
        indices = np.searchsorted(cum_weights, r + np.arange(N) / N)
        indices = np.clip(indices, 0, N - 1)
        resampled = updated_ensemble[indices]
        # Add small perturbation to avoid identical members
        ens_spread = np.std(updated_ensemble, axis=0)
        perturbation_scale = 0.05 * ens_spread
        resampled = resampled + np.random.randn(N, D) * perturbation_scale
        updated_ensemble = resampled

    # Deadline guard
    if time.time() > deadline:
        updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
        updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
        return updated_ensemble

    # Final safety checks
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)

    return updated_ensemble

# EVOLVE-BLOCK-END
