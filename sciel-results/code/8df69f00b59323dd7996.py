"""Lorenz-96 assimilation using a localized Ensemble Kalman Filter with adaptive inflation."""
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

def _local_ekf_update(ens, observations, observed_indices, observation_variance, localization_decay=3.0):
    """
    Single localized EnKF update step.
    
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
    
    # Compute ensemble mean and covariance
    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Observation error covariance
    R = np.eye(n_obs) * observation_variance
    
    # Get observed state components from ensemble
    ens_obs = ens[:, observed_indices]  # (n_members, n_obs)
    ens_obs_mean = np.mean(ens_obs, axis=0)
    
    # Compute innovation
    innovation = observations - ens_obs_mean
    
    # Compute innovation covariance S = H P H^T + R
    P_obs_obs = cov[np.ix_(observed_indices, observed_indices)]
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
    # This is crucial to avoid spurious correlations in small ensembles
    localized_K = np.zeros_like(K)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            # Rational localization: w = 1/(1 + (d/L)^2)
            w = 1.0 / (1.0 + (d / localization_decay) ** 2)
            localized_K[i, j] = K[i, j] * w
    
    # Update each ensemble member
    # Perturbed observations approach: each member sees its own perturbed observation
    # This preserves the ensemble spread better than using the mean innovation
    # We generate perturbations for each member
    rng = np.random.default_rng()
    # We need to ensure reproducibility or just use standard normal
    # For EnKF, it's common to use the same perturbation for all members in a step if we want to be deterministic,
    # but here we are just updating the state. The standard EnKF update is:
    # x_a = x_f + K (y - H x_f)
    # For ensemble members:
    # x_a_i = x_f_i + K (y_i - H x_f_i)
    # where y_i = y + e_i, e_i ~ N(0, R)
    # However, a simpler and often more stable approach for small ensembles is the "mean update" 
    # or "perturbed observation" method.
    # Let's use the perturbed observation method to maintain spread.
    
    # Generate independent perturbations for each member
    # n_members x n_obs
    perturbations = rng.normal(0, np.sqrt(observation_variance), size=(n_members, n_obs))
    member_observations = observations[None, :] + perturbations
    
    # Compute member innovations
    member_innovations = member_observations - ens_obs  # (n_members, n_obs)
    
    # Apply update
    update = member_innovations @ localized_K.T  # (n_members, n_state)
    ens_updated = ens + update
    
    return ens_updated

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform localized Ensemble Kalman Filter analysis update.
    
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
    
    # 1. Apply Localized EnKF Update
    ens = _local_ekf_update(ens, observations, observed_indices, observation_variance, localization_decay=3.0)
    
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
