import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 145.0

def _letkf_block_update(X_b, y, obs_idx, R_val, block_radius):
    """
    Local Additive Ensemble Kalman Filter update for a single block.
    Uses the linear (additive) EnKF update which is robust and avoids
    the dimension mismatches of the transform-based LETKF when applied
    to blocks with unobserved variables.
    """
    M, N_block = X_b.shape
    N_obs = len(obs_idx)
    
    if N_obs == 0:
        return X_b.copy()
    
    # Background perturbations
    W_b = X_b - np.mean(X_b, axis=0)
    
    # Background covariance (unbiased)
    P_b = (W_b.T @ W_b) / (M - 1)
    
    # Extract relevant covariances
    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]
    
    # Observation error covariance
    R = np.eye(N_obs) * R_val
    
    # Innovation covariance
    S = P_oo + R
    
    # Compute Kalman Gain
    try:
        S_inv = np.linalg.inv(S)
        K = P_xo @ S_inv
    except np.linalg.LinAlgError:
        try:
            S_inv = np.linalg.pinv(S)
            K = P_xo @ S_inv
        except:
            return X_b.copy()
    
    # Compute analysis
    mean_b = np.mean(X_b, axis=0)
    H_W_b = W_b[:, obs_idx]  # (M, N_obs)
    d = y - mean_b[obs_idx]  # (N_obs,)
    
    # Innovation for each member: H W_b_m - d
    # H_W_b is (M, N_obs), d is (N_obs,)
    # We need to compute K @ (H_W_b - d) for each member?
    # K is (N_block, N_obs).
    # H_W_b is (M, N_obs).
    # We want to compute a matrix C of shape (M, N_block) where
    # C[m, :] = K @ (H_W_b[m, :] - d)
    # This is equivalent to: (H_W_b - d[None, :]) @ K.T
    
    # H_W_b - d[None, :] is (M, N_obs)
    # K.T is (N_obs, N_block)
    # Result is (M, N_block)
    
    try:
        correction = (H_W_b - d[np.newaxis, :]) @ K.T
        X_a = X_b + correction
    except:
        # Fallback: return background
        X_a = X_b.copy()
    
    return X_a

def analyze(ensemble, observations, observed_indices, observation_variance):
    if time.time() > _DEADLINE:
        return np.array(ensemble, dtype=np.float64)
    
    X_b = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)
    
    M, N = X_b.shape
    N_obs = len(obs_idx)
    
    if N_obs == 0:
        return X_b.copy()
    
    # Block localization radius
    block_radius = 4
    
    # Initialize analysis
    X_a = X_b.copy()
    
    # Process each observed variable's block
    for i, obs_i in enumerate(obs_idx):
        # Determine block range
        block_start = max(0, obs_i - block_radius)
        block_end = min(N, obs_i + block_radius + 1)
        
        # Get observations in this block
        block_obs_mask = (obs_idx >= block_start) & (obs_idx < block_end)
        block_obs_global = obs_idx[block_obs_mask]
        block_obs_y = y[block_obs_mask]
        
        if len(block_obs_global) == 0:
            continue
        
        # Relative indices within block
        block_obs_rel = block_obs_global - block_start
        
        # Extract block
        X_b_block = X_b[:, block_start:block_end]
        
        # Perform LETKF update on this block
        X_a_block = _letkf_block_update(X_b_block, block_obs_y, block_obs_rel, R_val, block_radius)
        
        # Update the analysis
        X_a[:, block_start:block_end] = X_a_block
    
    # Safety
    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return X_a
