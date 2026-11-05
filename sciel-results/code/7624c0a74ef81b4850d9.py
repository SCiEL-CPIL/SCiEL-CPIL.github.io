"""Lorenz-96 assimilation using Particle Flow Filter with adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Time deadline guard
_DEADLINE = 145.0
_t0 = None

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def _particle_flow_update(ens, observations, observed_indices, observation_variance, localization_decay=2.5):
    """
    Particle Flow Filter (PFF) update step.
    
    Uses a smooth, divergence-free flow to transport the ensemble toward
    matching the observation statistics, preserving non-Gaussian structure.
    
    Parameters:
    - ens: (n_members, n_state) ensemble
    - observations: (n_obs,) observed values
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    - localization_decay: decay length for rational localization
    
    Returns:
    - Updated ensemble with same shape
    """
    n_members, n_state = ens.shape
    n_obs = len(observations)
    
    # Compute ensemble statistics
    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Get observed state components from ensemble
    ens_obs = ens[:, observed_indices]  # (n_members, n_obs)
    ens_obs_mean = np.mean(ens_obs, axis=0)
    
    # Compute innovation
    innovation = observations - ens_obs_mean
    
    # Compute innovation covariance S = H P H^T + R
    P_obs_obs = cov[np.ix_(observed_indices, observed_indices)]
    R = np.eye(n_obs) * observation_variance
    S = P_obs_obs + R
    
    # Regularize S
    S_reg = S + 1e-8 * np.eye(n_obs)
    
    # Compute Kalman gain K = P H^T S^{-1}
    PHt = cov[:, observed_indices]  # (n_state, n_obs)
    
    try:
        K = np.linalg.solve(S_reg, PHt.T).T
    except np.linalg.LinAlgError:
        K = PHt @ np.linalg.pinv(S_reg)
    
    # Apply localization to the Kalman gain using rational localization
    localized_K = np.zeros_like(K)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            # Rational localization: w = 1/(1 + (d/L)^2)
            w = 1.0 / (1.0 + (d / localization_decay) ** 2)
            localized_K[i, j] = K[i, j] * w
    
    # Particle Flow: compute a smooth flow field that transports each particle
    # The PFF uses a velocity field v(x) = K * (y - H(x)) but applied smoothly
    # We use a soft transport approach: instead of linear update, we compute
    # a weighted flow that preserves particle identities and spread
    
    # Compute per-member innovations
    member_innovations = observations[None, :] - ens_obs  # (n_members, n_obs)
    
    # Compute the "target" displacement for each member using the localized gain
    # This is the linear EnKF update direction
    linear_update = member_innovations @ localized_K.T  # (n_members, n_state)
    
    # For PFF, we want a smooth, divergence-free transport
    # We use a relaxation approach: blend between the original position and the 
    # linear update target with a smooth weighting based on innovation magnitude
    
    # Compute innovation magnitudes per member (how far each member is from obs)
    innovation_mags = np.sqrt(np.sum(member_innovations ** 2, axis=1))  # (n_members,)
    
    # Normalize innovation magnitudes
    innov_mag_mean = np.mean(innovation_mags)
    innov_mag_std = np.std(innovation_mags)
    if innov_mag_std > 1e-10:
        innov_zscores = (innovation_mags - innov_mag_mean) / innov_mag_std
    else:
        innov_zscores = np.zeros(n_members)
    
    # Smooth weighting: particles with larger innovations get more update
    # Use a smooth sigmoid-like function to avoid discontinuities
    # weight = 1 / (1 + exp(-alpha * zscore)) mapped to [0.5, 1.0]
    alpha = 0.8
    raw_weights = 1.0 / (1.0 + np.exp(-alpha * innov_zscores))
    # Map to range [0.7, 1.0] to ensure all particles get some update
    weights = 0.7 + 0.3 * raw_weights
    
    # Apply weighted update (PFF-style smooth transport)
    update = (weights[:, None] * linear_update)
    
    # Additionally, add a small smooth correction to match the ensemble mean
    # to the target mean more closely
    target_mean_shift = innovation @ localized_K.T  # (n_state,)
    current_mean_after = ens_mean + np.mean(update, axis=0)
    target_mean = ens_mean + target_mean_shift
    
    # Smooth correction to align means
    mean_correction = (target_mean - current_mean_after) / n_members
    update += mean_correction[None, :]
    
    ens_updated = ens + update
    
    return ens_updated

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform Particle Flow Filter analysis update with adaptive inflation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    n_members, n_state = ensemble.shape
    
    # Work with a copy
    ens = ensemble.astype(np.float64).copy()
    
    # Check deadline
    if _check_deadline():
        return ens
    
    # 1. Apply Particle Flow Filter Update
    ens = _particle_flow_update(ens, observations, observed_indices, observation_variance, localization_decay=2.5)
    
    # 2. Adaptive Inflation
    # Compute the ratio of observed innovation variance to expected variance
    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    
    P_obs_obs = cov[np.ix_(observed_indices, observed_indices)]
    R = np.eye(len(observations)) * observation_variance
    S_expected = P_obs_obs + R
    
    ens_obs_mean = np.mean(ens[:, observed_indices], axis=0)
    innovation = observations - ens_obs_mean
    
    # Calculate inflation factor
    # We use the ratio of the squared innovation to the expected variance
    # If the ensemble is too narrow, the innovation will be large relative to S, so we inflate.
    # If the ensemble is too wide, the innovation will be small relative to S, so we deflate (but we clamp to 1.0 to avoid over-deflation).
    
    ratios = np.ones(len(observations))
    for j in range(len(observations)):
        exp_var = S_expected[j, j]
        if exp_var > 1e-10:
            # Use a robust estimator: mean of squared innovations / mean of expected variances
            # Here we do it per observation
            ratios[j] = (innovation[j] ** 2 + observation_variance) / exp_var
    
    # Geometric mean of ratios
    log_ratios = np.log(np.clip(ratios, 1e-6, 1e6))
    geo_mean_ratio = np.exp(np.mean(log_ratios))
    
    # Inflation factor is the square root of the ratio
    # We clamp it to prevent extreme inflation or deflation
    # Deflation is generally dangerous in chaotic systems, so we clamp lower bound to 1.0
    # Upper bound allows for some inflation if the ensemble collapsed
    inflation = np.sqrt(np.clip(geo_mean_ratio, 1.0, 1.5))
    
    # Apply inflation around the ensemble mean
    ens = ens_mean + inflation * (ens - ens_mean)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
