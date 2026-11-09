<<<<<<< SEARCH
"""Lorenz-96 ensemble assimilation with Local Ensemble Square Root Filter (LEsRF)."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Square Root Filter (LEsRF) analysis update for Lorenz-96.
    Uses sequential per-observation updates with Gaussian localization and
    adaptive multiplicative inflation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=np.int64)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # --- Adaptive Multiplicative Inflation ---
    ens_mean = np.mean(ens, axis=0)
    ens_anom = ens - ens_mean
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-12))
    target_spread = np.sqrt(R_val) * 1.2
    
    ratio = target_spread / np.maximum(current_spread, 1e-12)
    inflation_factor = np.clip(np.mean(ratio), 1.0, 1.15)
    inflation_factor = 1.0 + 0.6 * (inflation_factor - 1.0)
    inflation_factor = np.clip(inflation_factor, 1.0, 1.12)
    
    # Apply inflation
    ens = ens_mean + inflation_factor * ens_anom
    
    # --- Gaussian Localization ---
    sigma = 3.0
    indices = np.arange(n_vars)
    
    # Precompute localization weights for each observation
    loc_weights_all = np.zeros((n_obs, n_vars))
    for j in range(n_obs):
        dist = np.abs(indices - obs_idx[j])
        dist = np.minimum(dist, n_vars - dist)
        loc_weights_all[j] = np.exp(-0.5 * (dist / sigma) ** 2)
    
    # --- Sequential LEsRF updates ---
    # Interleave observations for better mixing
    order = np.arange(n_obs)
    if n_obs > 1:
        even = order[order % 2 == 0]
        odd = order[order % 2 == 1]
        interleaved = np.empty(n_obs, dtype=np.int64)
        pos = 0
        ei, oi = 0, 0
        while ei < len(even) or oi < len(odd):
            if ei < len(even):
                interleaved[pos] = even[ei]; ei += 1; pos += 1
            if oi < len(odd):
                interleaved[pos] = odd[oi]; oi += 1; pos += 1
        order = interleaved
    
    for j in order:
        if time.time() > deadline:
            break
        
        obs_j = obs[j]
        obs_loc = obs_idx[j]
        w = loc_weights_all[j]  # (n_vars,)
        
        # Current ensemble statistics
        ens_mean = np.mean(ens, axis=0)
        x_obs = ens[:, obs_loc]
        x_mean = np.mean(x_obs)
        x_anom = x_obs - x_mean
        
        # Localized ensemble variance at obs location
        P_oo = np.var(x_obs, ddof=1)
        S = P_oo + R_val
        
        # Innovation
        innovation = obs_j - x_mean
        
        # Localized cross-covariance for each variable
        ens_anom_all = ens - ens_mean  # (n_members, n_vars)
        P_xo = np.cov(ens_anom_all, x_obs, rowvar=False, bias=False) * w  # (n_vars,)
        
        # Gain for each variable
        K = P_xo / S  # (n_vars,)
        
        # Square-root LEsRF update:
        # x_a^i = x_f^i + K * (y - x_f^i)  [standard EnKF form]
        # This gives correct mean; spread is slightly reduced but inflation compensates.
        ens = ens + K[None, :] * (innovation - x_anom)[None, :]
    
    # --- Post-processing ---
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
=======
"""Lorenz-96 ensemble assimilation with Local Ensemble Square Root Filter (LEsRF)."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Square Root Filter (LEsRF) analysis update for Lorenz-96.
    Uses sequential per-observation updates with Gaussian localization and
    adaptive multiplicative inflation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=np.int64)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # --- Adaptive Multiplicative Inflation ---
    ens_mean = np.mean(ens, axis=0)
    ens_anom = ens - ens_mean
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-12))
    target_spread = np.sqrt(R_val) * 1.2
    
    ratio = target_spread / np.maximum(current_spread, 1e-12)
    inflation_factor = np.clip(np.mean(ratio), 1.0, 1.15)
    inflation_factor = 1.0 + 0.6 * (inflation_factor - 1.0)
    inflation_factor = np.clip(inflation_factor, 1.0, 1.12)
    
    # Apply inflation
    ens = ens_mean + inflation_factor * ens_anom
    
    # --- Gaussian Localization ---
    sigma = 3.0
    indices = np.arange(n_vars)
    
    # Precompute localization weights for each observation
    loc_weights_all = np.zeros((n_obs, n_vars))
    for j in range(n_obs):
        dist = np.abs(indices - obs_idx[j])
        dist = np.minimum(dist, n_vars - dist)
        loc_weights_all[j] = np.exp(-0.5 * (dist / sigma) ** 2)
    
    # --- Sequential LEsRF updates ---
    # Interleave observations for better mixing
    order = np.arange(n_obs)
    if n_obs > 1:
        even = order[order % 2 == 0]
        odd = order[order % 2 == 1]
        interleaved = np.empty(n_obs, dtype=np.int64)
        pos = 0
        ei, oi = 0, 0
        while ei < len(even) or oi < len(odd):
            if ei < len(even):
                interleaved[pos] = even[ei]; ei += 1; pos += 1
            if oi < len(odd):
                interleaved[pos] = odd[oi]; oi += 1; pos += 1
        order = interleaved
    
    for j in order:
        if time.time() > deadline:
            break
        
        obs_j = obs[j]
        obs_loc = obs_idx[j]
        w = loc_weights_all[j]  # (n_vars,)
        
        # Current ensemble statistics
        ens_mean = np.mean(ens, axis=0)
        x_obs = ens[:, obs_loc]
        x_mean = np.mean(x_obs)
        x_anom = x_obs - x_mean
        
        # Localized ensemble variance at obs location
        P_oo = np.var(x_obs, ddof=1)
        S = P_oo + R_val
        
        # Innovation
        innovation = obs_j - x_mean
        
        # Localized cross-covariance for each variable
        ens_anom_all = ens - ens_mean  # (n_members, n_vars)
        P_xo = np.cov(ens_anom_all, x_obs, rowvar=False, bias=False) * w  # (n_vars,)
        
        # Gain for each variable
        K = P_xo / S  # (n_vars,)
        
        # Standard EnKF update: x_a = x_f + K*(y - x_f)
        ens = ens + K[None, :] * (innovation - x_anom)[None, :]
    
    # --- Post-processing ---
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
