"""Lorenz-96 assimilation using a localized Ensemble Kalman Filter with adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Time deadline guard
_DEADLINE = 140.0
_t0 = None

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a localized Ensemble Kalman Filter analysis update with adaptive inflation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Work with a copy
    ens = ensemble.astype(np.float64).copy()
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean
    
    # Sample covariance (N-1) with small regularization for conditioning
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    cov += 1e-6 * np.eye(n_state)
    
    # Observation operator: identity for observed indices
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation error covariance
    R = np.eye(n_obs) * observation_variance
    
    # Innovation
    innovation = observations - H @ ens_mean
    
    # Innovation covariance S = H P H^T + R
    HP = H @ cov
    S = HP @ H.T + R
    
    # Regularize S for numerical stability
    S_reg = S + 1e-8 * np.eye(n_obs)
    
    # Kalman gain K = P H^T S^{-1} using solve for better stability
    PHt = cov @ H.T  # (n_state, n_obs)
    
    try:
        # Solve S_reg X = PHt^T, then transpose
        K = np.linalg.solve(S_reg, PHt.T).T
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S_reg)
        K = PHt @ S_inv
    
    # Apply spatial localization to the Kalman gain
    # Sharper decay to reduce spurious correlations
    decay_length = 1.8
    
    localized_K = np.zeros_like(K)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            w = np.exp(-d ** 2 / (2.0 * decay_length ** 2))
            localized_K[i, j] = K[i, j] * w
    
    # Ensemble update: x_m^a = x_m^f + K_loc * (y - H x_m^f)
    Hx = ens[:, observed_indices]  # (n_members, n_obs)
    
    # Innovation for each member
    member_innovations = observations[None, :] - Hx  # (n_members, n_obs)
    
    # Update: ens_a = ens_f + localized_K @ member_innovations^T
    update = member_innovations @ localized_K.T  # (n_members, n_state)
    ens = ens + update
    
    # Adaptive inflation: per-observation local spread matching
    ens_mean_new = np.mean(ens, axis=0)
    anomalies_new = ens - ens_mean_new
    cov_new = (anomalies_new.T @ anomalies_new) / (n_members - 1)
    
    # Compute per-observation expected innovation variance
    HP_new = H @ cov_new
    S_expected = HP_new @ H.T + R
    S_expected_diag = np.diag(S_expected)
    
    # Per-observation actual innovation
    innov_per_obs = innovation  # (n_obs,)
    
    # Inflation factor: blend between 1.0 and a spread-matching factor
    # Use local ratio for each observation, then take geometric mean
    ratios = np.ones(n_obs)
    for j in range(n_obs):
        exp_var = S_expected_diag[j]
        act_var = innov_per_obs[j] ** 2 + observation_variance  # include obs noise
        if exp_var > 1e-10 and act_var > 1e-10:
            ratios[j] = act_var / exp_var
    
    # Geometric mean of ratios, clipped
    log_ratios = np.log(np.clip(ratios, 1e-6, 1e6))
    geo_mean_ratio = np.exp(np.mean(log_ratios))
    inflation = np.sqrt(np.clip(geo_mean_ratio, 0.9, 1.3))
    
    # Apply inflation around the new mean
    ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
