"""Lorenz-96 assimilation using Ensemble Square Root Filter with adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE = 145.0
_t0 = None

def _check_deadline():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Square Root Filter analysis update for Lorenz-96.
    
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
    
    ens = ensemble.astype(np.float64).copy()
    obs_idx = np.asarray(observed_indices)
    
    # Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # Precompute localization weights for each observed index
    # Use exponential decay with circular distance
    decay_len = 2.0
    loc_weights = np.zeros((n_obs, n_state))
    for j, oidx in enumerate(obs_idx):
        for i in range(n_state):
            d = abs(i - oidx)
            d = min(d, n_state - d)
            loc_weights[j, i] = np.exp(-d / decay_len)
    
    # Sequential EnSRF: process observations one at a time
    # For each observation, regress unobserved variables onto it
    for j in range(n_obs):
        oidx = obs_idx[j]
        obs_val = observations[j]
        obs_var = observation_variance
        
        # Current ensemble mean
        cur_mean = np.mean(ens, axis=0)
        
        # Ensemble anomalies for this observation coordinate
        x_obs = ens[:, oidx]  # (n_members,)
        x_obs_mean = np.mean(x_obs)
        
        # Innovation
        innov = obs_val - x_obs_mean
        
        # Variance of observed variable in ensemble
        var_obs = np.var(x_obs, ddof=1)
        
        # Avoid division by zero
        if var_obs < 1e-10:
            continue
        
        # Kalman gain for this single observation
        # For unobserved coordinates i: K_i = cov(x_i, x_obs) / (var_obs + R)
        # cov(x_i, x_obs) = (1/(n-1)) * sum_m (x_i_m - mean_i)(x_obs_m - mean_obs)
        
        denom = var_obs + obs_var
        
        # Update all coordinates
        for i in range(n_state):
            if i == oidx:
                # For the observed coordinate, use standard EKF update
                # But we need to preserve the mean shift
                x_i = ens[:, i]
                x_i_mean = np.mean(x_i)
                cov_io = np.cov(x_i, x_obs, ddof=1)[0, 1]
                K = cov_io / denom
                # Update: x_i_a = x_i_f + K * (obs - H*x_mean)
                # For observed coord: x_a = x_f + K*(obs - x_mean)
                # But we want the new mean to be closer to obs
                # New mean = x_mean + K*(obs - x_mean) = (1-K)*x_mean + K*obs
                # Anomalies scaled by (1-K)
                new_mean = x_i_mean + K * innov
                new_anoms = (1 - K) * (x_i - x_i_mean)
                ens[:, i] = new_mean + new_anoms
            else:
                # For unobserved coordinates, apply localization
                w = loc_weights[j, i]
                if w < 0.01:
                    continue
                x_i = ens[:, i]
                x_i_mean = np.mean(x_i)
                cov_io = np.cov(x_i, x_obs, ddof=1)[0, 1]
                K = w * cov_io / denom
                new_mean = x_i_mean + K * innov
                new_anoms = (1 - K) * (x_i - x_i_mean)
                ens[:, i] = new_mean + new_anoms
    
    # Adaptive inflation: check if ensemble spread is consistent with observation noise
    # Compute expected innovation variance from updated ensemble
    ens_mean_new = np.mean(ens, axis=0)
    
    # For each observed coordinate, compute ensemble variance
    spread_ok = True
    total_ratio = 0.0
    count = 0
    for j, oidx in enumerate(obs_idx):
        x_obs_new = ens[:, oidx]
        var_new = np.var(x_obs_new, ddof=1)
        # Expected innovation variance should be ~ obs_var
        # If ensemble spread is too small, inflate
        if var_new < 1e-10:
            spread_ok = False
            break
        ratio = obs_var / var_new
        total_ratio += ratio
        count += 1
    
    if count > 0:
        mean_ratio = total_ratio / count
        # Inflate if spread is too small (ratio > 1), deflate slightly if too large
        inflation = np.sqrt(np.clip(mean_ratio, 0.85, 1.6))
        inflation = np.clip(inflation, 0.95, 1.4)
        
        # Apply inflation around mean
        ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Final safety: clip and ensure finite
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
