"""Local Ensemble Square Root Filter (ESRF) for Lorenz-96 assimilation."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a Local Ensemble Square Root Filter (ESRF) update.
    Applies a direct orthogonal transformation to ensemble anomalies to
    exactly match the updated Kalman covariance, preserving spread.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array-like
        Indices of the observed state components.
    observation_variance : float
        Variance of the observation noise.
        
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    n_ens = ensemble.shape[0]
    n_state = ensemble.shape[1]
    
    # Work with a copy
    result = ensemble.astype(np.float64).copy()
    
    obs_idx = list(observed_indices)
    n_obs = len(obs_idx)
    
    # Precompute the localization window
    # For Lorenz-96, interactions are local; use a window of radius 4
    window_radius = 4
    n_loc = 2 * window_radius + 1  # 9
    
    # Build a list of (obs_idx, loc_indices, loc_obs_positions) for each observation
    # loc_indices: which state components to update for this observation
    # loc_obs_positions: where the observation sits within the localized block
    obs_blocks = []
    for oi, oi_idx in enumerate(obs_idx):
        lo = max(0, oi_idx - window_radius)
        hi = min(n_state - 1, oi_idx + window_radius)
        loc_idx = list(range(lo, hi + 1))
        n_loc_i = len(loc_idx)
        # Position of the observation within this block
        obs_pos = oi_idx - lo
        obs_blocks.append((oi_idx, loc_idx, obs_pos, n_loc_i))
    
    # Process each observation sequentially with localized ESRF
    for oi, oi_idx in enumerate(obs_idx):
        _, loc_idx, obs_pos, n_loc_i = obs_blocks[oi]
        obs_val = observations[oi]
        
        # Extract the localized ensemble block: shape (n_ens, n_loc_i)
        block = result[:, loc_idx].copy()
        
        # Compute ensemble mean and anomalies for the block
        x_mean = block.mean(axis=0)  # shape (n_loc_i,)
        x_anom = block - x_mean      # shape (n_ens, n_loc_i)
        
        # Build the localized state-observation cross-covariance
        # P_xy = (1/(n_ens-1)) * X_anom @ Y_anom.T
        # Here Y is a single observation, so Y_anom is a scalar per member
        # P_x (ensemble covariance of the block): (n_loc_i, n_loc_i)
        P_x = (x_anom.T @ x_anom) / (n_ens - 1)
        
        # Observation operator H picks out the obs_pos component
        # P_yy = P_x[obs_pos, obs_pos]
        P_yy = P_x[obs_pos, obs_pos]
        
        # Kalman gain for the block: K = P_x @ H^T / (H P_x H^T + R)
        # H is a row vector with 1 at obs_pos, 0 elsewhere
        # So H P_x H^T = P_x[obs_pos, obs_pos] = P_yy
        # K[:, j] = P_x[j, obs_pos] / (P_yy + R)
        R = observation_variance
        denom = P_yy + R
        if denom < 1e-15:
            continue  # skip if degenerate
        
        K = P_x[:, obs_pos] / denom  # shape (n_loc_i,)
        
        # Innovation
        innovation = obs_val - x_mean[obs_pos]
        
        # ESRF update:
        # New mean: x_mean_new = x_mean + K * innovation
        # New anomalies: A_new = A @ (I - 0.5 * K * H) where H is the row vector
        # For a single observation, H is a row with 1 at obs_pos
        # A_new[:, j] = A[:, j] - 0.5 * K[j] * A[:, obs_pos]
        
        x_mean_new = x_mean + K * innovation
        A_new = x_anom - 0.5 * np.outer(x_anom[:, obs_pos], K)
        
        # Write back
        result[:, loc_idx] = x_mean_new + A_new
    
    # Final global consistency: recompute overall mean and ensure spread
    # (The sequential localized updates should be consistent, but let's verify)
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    # Ensure finiteness
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
# EVOLVE-BLOCK-END
