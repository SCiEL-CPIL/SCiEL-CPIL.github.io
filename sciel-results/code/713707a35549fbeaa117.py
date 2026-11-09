"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perturbed Ensemble Kalman Filter update with localization and adaptive inflation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of observed_indices
    - observed_indices: array of indices that were observed (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    start_time = time.time()
    
    n_members, n_vars = ensemble.shape
    ens = ensemble.astype(np.float64).copy()
    
    # Determine which indices are observed and unobserved
    obs_idx = np.array(sorted(observed_indices))
    all_idx = np.arange(n_vars)
    unobs_idx = np.array([i for i in all_idx if i not in set(obs_idx.tolist())])
    
    # Compute ensemble mean
    mean = ens.mean(axis=0)
    
    # Compute centered anomalies
    anomalies = ens - mean  # shape (n_members, n_vars)
    
    # Compute covariance matrix
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Localize covariance using exponential decay on ring distance
    decay_scale = 2.5
    
    def ring_distance(i, j):
        d = abs(i - j)
        return min(d, n_vars - d)
    
    # Build localization weight matrix for all pairs
    def localize_block(block, idx_rows, idx_cols):
        """Apply localization to a covariance block."""
        localized = block.copy()
        for ii in range(len(idx_rows)):
            for jj in range(len(idx_cols)):
                d = ring_distance(idx_rows[ii], idx_cols[jj])
                localized[ii, jj] *= np.exp(-0.5 * (d / decay_scale) ** 2)
        return localized
    
    # Localize observed-observed and unobserved-observed blocks
    cov_oo = localize_block(cov[np.ix_(obs_idx, obs_idx)], obs_idx, obs_idx)
    cov_uo = localize_block(cov[np.ix_(unobs_idx, obs_idx)], unobs_idx, obs_idx)
    
    # Observation noise covariance (diagonal)
    R = observation_variance * np.eye(len(obs_idx))
    
    # Innovation covariance: S = H P H^T + R
    S = cov_oo + R
    
    # Compute Kalman gain using Cholesky decomposition for stability
    try:
        L = np.linalg.cholesky(S)
        S_inv = np.linalg.inv(np.linalg.cholesky(S).T @ np.linalg.cholesky(S))
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Innovation
    innovation = observations - mean[obs_idx]
    
    # Compute full Kalman gain K (n_vars x n_obs)
    K_full = np.zeros((n_vars, len(obs_idx)))
    K_full[obs_idx, :] = cov_oo @ S_inv
    K_full[unobs_idx, :] = cov_uo.T @ S_inv
    
    # Perturbed ensemble update:
    # For each member, compute its own innovation and apply member-specific update
    # This preserves ensemble spread
    updated_ens = np.zeros_like(ens)
    
    for m in range(n_members):
        # Member-specific innovation
        member_mean_obs = ens[m, obs_idx]
        member_innovation = observations - member_mean_obs
        
        # Member-specific update
        member_update = K_full @ member_innovation
        
        # Apply update to this member
        updated_ens[m, :] = ens[m, :] + member_update
    
    # Adaptive inflation: monitor spread and inflate if too small
    new_mean = updated_ens.mean(axis=0)
    new_spread = updated_ens.std(axis=0)
    old_spread = ens.std(axis=0)
    
    # Target spread: slightly larger than pre-update spread
    target_spread = old_spread * 1.01
    
    # Inflation factor per variable
    inflation = np.ones(n_vars)
    for j in range(n_vars):
        if new_spread[j] < target_spread[j] * 0.8:
            # Need to inflate
            if new_spread[j] > 1e-10:
                inflation[j] = target_spread[j] / new_spread[j]
            else:
                inflation[j] = 1.05
    
    # Apply inflation around the new mean
    for j in range(n_vars):
        if inflation[j] > 1.0:
            updated_ens[:, j] = new_mean[j] + (updated_ens[:, j] - new_mean[j]) * inflation[j]
    
    # Check for non-finite values
    if not np.all(np.isfinite(updated_ens)):
        updated_ens = np.nan_to_num(updated_ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Clip to valid range
    updated_ens = np.clip(updated_ens, -100.0, 100.0)
    
    return updated_ens

# EVOLVE-BLOCK-END
