"""Ensemble Kalman Smoother with 2-step back-window for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gaspari_cohn_weights(center, N, half_window, c):
    """
    Compute Gaspari-Cohn localization weights for a local window centered at 'center'.
    Returns an array of shape (2*half_window+1,) with weights for each position.
    """
    indices = [(center - half_window + j) % N for j in range(2 * half_window + 1)]
    distances = np.array([min(abs(i - center), N - abs(i - center)) for i in indices], dtype=float)
    r = distances / c
    result = np.zeros_like(r)
    
    # r <= 0.5
    m1 = r <= 0.5
    r1 = r[m1]
    result[m1] = (-5.0/4.0)*r1**5 + 5.0*r1**4 - 10.0*r1**3 + 5.0*r1
    
    # 0.5 < r <= 1.5
    m2 = (r > 0.5) & (r <= 1.5)
    r2 = r[m2]
    result[m2] = (1.0/12.0)*r2**5 - (5.0/4.0)*r2**4 + (5.0/2.0)*r2**3 - 5.0*r2**2 + (25.0/12.0)*r2 + 0.5
    
    # 1.5 < r <= 2.5
    m3 = (r > 1.5) & (r <= 2.5)
    r3 = r[m3]
    result[m3] = (-1.0/16.0)*r3**5 + (1.0/4.0)*r3**4 - (15.0/8.0)*r3**3 + (5.0/2.0)*r3**2 - (25.0/12.0)*r3 + 105.0/48.0
    
    # Ensure center weight is 1
    result[half_window] = 1.0
    
    return result


def _local_ens_update(ens, obs_idx, obs_val, obs_var, N, window_size=3, corr_len=4.0):
    """
    Perform a local ensemble update for a single observed variable with Gaspari-Cohn localization.
    Updates the local window around obs_idx.
    """
    M = ens.shape[0]
    i = obs_idx
    
    # Local indices with periodic boundary conditions
    local_indices = [(i - window_size + j) % N for j in range(2 * window_size + 1)]
    
    # Extract local ensemble (M x (2*window_size+1))
    local_ens = ens[:, local_indices].copy()
    
    # The observed variable is at the center of the window
    center_idx = window_size
    
    # Compute local mean and anomalies
    local_mean = local_ens.mean(axis=0)
    local_anom = local_ens - local_mean
    
    # Apply Gaspari-Cohn localization to anomalies
    w = _gaspari_cohn_weights(i, N, window_size, corr_len)
    local_anom_loc = local_anom * w[np.newaxis, :]
    
    # Local covariance with localization
    local_cov = (local_anom_loc.T @ local_anom_loc) / (M - 1)
    
    # Observation variance
    R = obs_var
    
    # Innovation covariance S = local_cov[center, center] + R
    S = local_cov[center_idx, center_idx] + R
    
    # Ensure S is positive
    if S < 1e-10:
        S = 1e-10
    
    # Gain vector K (2*window_size+1 x 1)
    # K = local_cov[:, center_idx] / S
    K = local_cov[:, center_idx] / S
    
    # Innovation for each ensemble member
    innovations = obs_val - local_ens[:, center_idx]
    
    # Update local ensemble
    local_ens_updated = local_ens + np.outer(innovations, K)
    
    # Write back to full ensemble
    ens[:, local_indices] = local_ens_updated
    
    return ens


def _apply_smoothing(ens_current, ens_prev, obs_idx, obs_val, obs_var, N, 
                     window_size=3, corr_len=4.0, smooth_weight=0.3):
    """
    Apply 2-step back-window smoothing: update both current and previous states.
    The previous state is updated with a damped version of the innovation to exploit
    temporal correlation and correct for propagation errors.
    """
    M = ens_current.shape[0]
    i = obs_idx
    
    # Update current state fully
    ens_current = _local_ens_update(ens_current, i, obs_val, obs_var, N, window_size, corr_len)
    
    # Update previous state with damped innovation (smoothing)
    # The previous state should be less aggressive since it's already propagated
    damped_var = obs_var / (1.0 - smooth_weight)  # Effective observation error for previous state
    ens_prev = _local_ens_update(ens_prev, i, obs_val, damped_var, N, window_size, corr_len)
    
    return ens_current, ens_prev


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Smoother with 2-step back-window for Lorenz-96.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    if K == 0:
        return ensemble.copy()
    
    # 1. Adaptive Inflation
    # Calculate spread at observed locations
    obs_std = np.sqrt(observation_variance)
    ens_spread = np.std(ensemble[:, observed_indices], axis=0)
    mean_spread = np.mean(ens_spread)
    
    # Ratio of ensemble spread to observation error
    ratio = mean_spread / obs_std if obs_std > 0 else 1.0
    
    # Inflation factor
    # If ratio is low, inflate more. If high, keep close to 1.
    # Target ratio ~ 1.0 - 1.5
    if ratio < 1.0:
        inflation = 1.0 + 0.15 * (1.0 - ratio)
    else:
        inflation = 1.0 + 0.03 * np.exp(-(ratio - 1.0))
    
    inflation = np.clip(inflation, 1.0, 1.3)
    
    # Apply multiplicative inflation
    mean = ensemble.mean(axis=0)
    inflated = mean + inflation * (ensemble - mean)
    
    # 2. Apply local updates with 2-step smoothing
    # We'll apply updates sequentially, maintaining a smoothed previous state
    updated = inflated.copy()
    prev_state = inflated.copy()  # Initialize previous state
    
    window_size = 3  # Window of 7 variables
    corr_len = 4.0   # Correlation length for Gaspari-Cohn
    smooth_weight = 0.25  # Weight for smoothing the previous state
    
    for k in range(K):
        if time.time() > deadline:
            break
            
        idx = observed_indices[k]
        val = observations[k]
        var = observation_variance
        
        # Apply 2-step back-window smoothing
        updated, prev_state = _apply_smoothing(
            updated, prev_state, idx, val, var, N, 
            window_size, corr_len, smooth_weight
        )
        
        # Update previous state to be the current state for next iteration
        # This creates a sliding window effect
        prev_state = updated.copy()
        
    # 3. Safety Checks
    updated = np.clip(updated, -100.0, 100.0)
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated

# EVOLVE-BLOCK-END
