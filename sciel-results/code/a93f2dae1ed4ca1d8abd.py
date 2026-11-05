"""Lorenz-96 assimilation using sequential coordinate-wise Ensemble Square Root Filter with localization and adaptive inflation."""
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

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a sequential coordinate-wise Ensemble Square Root Filter analysis update
    with localization and adaptive inflation.
    
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
    
    # Localization decay length (tuned for Lorenz-96)
    decay_length = 2.0
    
    # Process observations sequentially (coordinate-wise)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        y_j = observations[j]
        
        # Get the current ensemble values at this observation index
        x_j = ens[:, obs_idx]  # (n_members,)
        x_j_mean = np.mean(x_j)
        x_j_anom = x_j - x_j_mean
        
        # Compute ensemble variance at this location
        x_j_var = np.var(x_j, ddof=1)
        
        # Innovation variance S = P_jj + R
        S = x_j_var + observation_variance
        if S < 1e-10:
            S = 1e-10
        
        # Kalman gain for this coordinate
        K_j = x_j_var / S
        
        # Apply update to all state components using localization
        for i in range(n_state):
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            w = np.exp(-d ** 2 / (2.0 * decay_length ** 2))
            
            # Covariance between state i and observed state j
            # Approximate using ensemble anomalies
            x_i_anom = ens[:, i] - np.mean(ens[:, i])
            P_ij = np.mean(x_i_anom * x_j_anom) / (n_members - 1)
            
            # Localized Kalman gain
            K_ij = w * P_ij / S
            
            # Update ensemble member i
            ens[:, i] += K_ij * (y_j - x_j)
        
        # Recompute x_j for next iteration (it has been updated)
        x_j = ens[:, obs_idx]
    
    # Adaptive inflation based on spread matching
    ens_mean_new = np.mean(ens, axis=0)
    anomalies_new = ens - ens_mean_new
    cov_new = (anomalies_new.T @ anomalies_new) / (n_members - 1)
    
    # Compute expected innovation variance for each observation
    ratios = np.ones(n_obs)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        P_jj = cov_new[obs_idx, obs_idx]
        S_expected = P_jj + observation_variance
        if S_expected > 1e-10:
            ratios[j] = observation_variance / S_expected
    
    log_ratios = np.log(np.clip(ratios, 1e-6, 1e6))
    geo_mean_ratio = np.exp(np.mean(log_ratios))
    inflation = np.sqrt(np.clip(geo_mean_ratio, 0.95, 1.25))
    
    # Apply inflation around the new mean
    ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
